import torch
from torch import nn


def get_embeddings(model, inputs, return_weights=False):
    for task_name in inputs['task_name']:
        assert task_name==inputs['task_name'][0],f"Examples in the same batch should come from the same task, " \
                                                f"but task {task_name} and task {inputs['task_name'][0]} are found"
    cur_results = {}
    gate_logits = {}
    for k in ['query', 'pos', 'neg']:
        cur_inputs = {
            'input_ids': inputs[f'{k}_input_ids'],
            'attention_mask': inputs[f'{k}_attention_mask'],
            'texts': inputs[f'{k}_texts'],
        }
        features = model(cur_inputs)
        cur_results[k] = features['sentence_embedding']

        if return_weights and 'gate_logits' in features:
            gate_logits[k] = features['gate_logits']

    embeddings_query = cur_results['query']
    embeddings_pos = cur_results['pos']
    embeddings_neg = cur_results['neg']

    if return_weights:
        return embeddings_query, embeddings_pos, embeddings_neg, gate_logits

    return embeddings_query, embeddings_pos, embeddings_neg


def get_guide_embeddings(model, inputs):
    for task_name in inputs['task_name']:
        assert task_name==inputs['task_name'][0],f"Examples in the same batch should come from the same task, " \
                                                f"but task {task_name} and task {inputs['task_name'][0]} are found"
    cur_results = {}
    for k in ['query', 'pos', 'neg']:
        cur_results[k] = model.encode(
            inputs[f'{k}_texts'],
            output_value="sentence_embedding",
            convert_to_tensor=True,
            batch_size=4,
        )

    embeddings_query = cur_results['query']
    embeddings_pos = cur_results['pos']
    embeddings_neg = cur_results['neg']

    return embeddings_query, embeddings_pos, embeddings_neg


def remove_diagonal(tensor):
    N = tensor.size(0)
    mask = ~torch.eye(N, dtype=bool)

    return tensor[mask].view(N, N-1)


def vectorized_compute_scores(args, anchor_embeddings, pos_embeddings, neg_embeddings, similarity_fct, index=False):
    pos_sim = similarity_fct(anchor_embeddings, pos_embeddings)

    # The next operation will result in a tensor containing the similarity score of the anchor
    # with all the negative samples.
    neg_sim = similarity_fct(anchor_embeddings.unsqueeze(1), neg_embeddings.unsqueeze(0))

    if index:
        neg_sim = remove_diagonal(neg_sim)

    scores = torch.cat([pos_sim.view(-1, 1), neg_sim], dim=1) / args.gist_cl_temperature

    return scores


def vectorized_contrastive_scores(args, query_embeddings, doc_embeddings, similarity_fct):
    sim = similarity_fct(query_embeddings, doc_embeddings)

    # Expanded contrastive partition function (Z) for enhanced contrastive loss.

    # anchor to other positive samples
    anchor_to_pos = remove_diagonal(
        similarity_fct(query_embeddings.unsqueeze(1), doc_embeddings.unsqueeze(0))
    )

    # pos document to other anchors
    pos_to_anchor = remove_diagonal(
        similarity_fct(doc_embeddings.unsqueeze(1), query_embeddings.unsqueeze(0))
    )

    # anchor to other anchors
    anchor_to_anchor = remove_diagonal(
        similarity_fct(query_embeddings.unsqueeze(1), query_embeddings.unsqueeze(0))
    )

    # pos document to other positive samples
    pos_to_pos = remove_diagonal(
        similarity_fct(doc_embeddings.unsqueeze(1), doc_embeddings.unsqueeze(0))
    )

    scores = torch.cat([sim.view(-1, 1), anchor_to_pos, pos_to_anchor, anchor_to_anchor, pos_to_pos], dim=1) / args.gist_cl_temperature

    return scores

# def vectorized_bidirectional_scores(args, anchor_embeddings, pos_embeddings, neg_embeddings, similarity_fct, index=None):
#     pos_sim = similarity_fct(anchor_embeddings, pos_embeddings)

#     # The next operation will result in a tensor containing the similarity score of the anchor
#     # with all the negative samples.
#     neg_sim = similarity_fct(anchor_embeddings.unsqueeze(1), neg_embeddings.unsqueeze(0))

#     if index is not None:
#         neg_sim = remove_diagonal(neg_sim)

#     scores = torch.cat([pos_sim.view(-1, 1), neg_sim], dim=1) / args.gist_cl_temperature

#     return scores


def compute_scores(args, num, anchor_emb, pos_emb, embeddings_neg, similarity_fct, index=None):
    # index: The index in the pair that we skip.
    # anchor_emb: (1, features) dimension
    # pos_emb: (1, features) dimension
    cur_score = similarity_fct(anchor_emb, pos_emb) / args.gist_cl_temperature

    for j in range(0, num):
        if index is not None:
            if j == index:
                continue

        one_neg_emb = embeddings_neg[j].unsqueeze(0)
        one_neg_score = similarity_fct(anchor_emb, one_neg_emb) / args.gist_cl_temperature
        cur_score = torch.cat([cur_score, one_neg_score], dim=-1)

    return cur_score


def compute_load_balancing_loss(gate_logits: torch.Tensor, top_k=2) -> float:
    aux_loss = 0

    if gate_logits is not None:
        # Compute the auxiliary load balancing loss.
        num_experts = gate_logits['query'].size(-1)

        aux_loss = [load_balancing_loss_func((gate_logits[k],), num_experts=num_experts, top_k=top_k) for k in ['query', 'pos', 'neg'] if k in gate_logits]
        # aux_loss = sum(aux_loss) / len(aux_loss)  # Mean of the losses.

        # We use the sum of the losses instead of the mean.
        # This is because the all the embeddings have passed through the gate
        # independently, and we want to penalize the case where the routing
        # between experts is too unbalanced across all the embeddings.
        aux_loss = sum(aux_loss)  # Sum of the losses.

    return aux_loss


def improved_in_batch_contrastive_loss(args, model, inputs, return_outputs=False):
    # TODO: Adjust the contrastive loss on whether the task is symmetric or asymmetric.
    gate_logits = None
    loss = 0

    if args.gist_router_aux_loss_coef > 0:
        embeddings_query, embeddings_pos, embeddings_neg, gate_logits = get_embeddings(model, inputs, return_weights=True)

        assert gate_logits is not None, "gate_logits should not be None if args.gist_router_aux_loss_coef > 0"
        aux_loss = compute_load_balancing_loss(gate_logits, top_k=args.gist_topk)

        loss += aux_loss * args.gist_router_aux_loss_coef
    else:
        embeddings_query, embeddings_pos, embeddings_neg = get_embeddings(model, inputs, return_weights=False)

    similarity_fct = nn.CosineSimilarity(dim=-1)

    # Compute the contrastive loss relative to the hard negatives.
    all_scores = vectorized_compute_scores(
        args=args,
        anchor_embeddings=embeddings_query,
        pos_embeddings=embeddings_pos,
        neg_embeddings=embeddings_neg,
        similarity_fct=similarity_fct,
        index=False,
    )

    labels = torch.zeros(all_scores.size(0)).long().to(embeddings_query.device)
    loss += nn.CrossEntropyLoss()(all_scores, labels)

    # Compute the contrastive loss relative to the in-batch positives and
    # permutation of queries.
    scores = vectorized_contrastive_scores(
        args=args,
        query_embeddings=embeddings_query,
        doc_embeddings=embeddings_pos,
        similarity_fct=similarity_fct,
    )

    labels = torch.zeros(scores.size(0)).long().to(embeddings_query.device)
    loss += nn.CrossEntropyLoss()(scores, labels)

    if return_outputs:
        outs = (loss, all_scores)
    else:
        outs = loss

    return outs


def orthogonal_loss(args, model, inputs, return_outputs=False):
    embeddings_query, embeddings_pos, embeddings_neg = get_embeddings(model, inputs, return_weights=False)

    similarity_fct = nn.CosineSimilarity(dim=-1)

    # Compute the cosine similarity between the queries and the positives.
    # This already takes into account the contrastive loss with respect to positive and other-query as negatives.
    query_pos_sim = similarity_fct(embeddings_query.unsqueeze(1), embeddings_pos.unsqueeze(0))

    # Subtract the margin.
    query_pos_sim -= args.gist_orthogonal_loss_margin
    eye_mask = torch.eye(query_pos_sim.size(0), dtype=bool).to(query_pos_sim.device)

    # Add the margin to the diagonal. Subtract the positive similarity from 1.
    query_pos_sim[eye_mask] = 1 - (query_pos_sim[eye_mask] + args.gist_orthogonal_loss_margin)

    # Set the negative values to 0.
    query_pos_sim[query_pos_sim < 0] = 0
    query_pos_sim = query_pos_sim.flatten()

    # Compute the cosine similarity between the query and the negative.
    neg_sim = similarity_fct(embeddings_query.unsqueeze(1), embeddings_neg.unsqueeze(0))
    neg_sim -= args.gist_orthogonal_loss_margin
    neg_sim[neg_sim < 0] = 0
    neg_sim = neg_sim.flatten()

    # Compute the cosine similarity between the positive and the negative.
    neg_pos_sim = similarity_fct(embeddings_pos.unsqueeze(1), embeddings_neg.unsqueeze(0))
    neg_pos_sim -= args.gist_orthogonal_loss_margin
    neg_pos_sim[neg_pos_sim < 0] = 0
    neg_pos_sim = neg_pos_sim.flatten()

    sims = torch.cat([query_pos_sim, neg_sim, neg_pos_sim], dim=0)

    # Compute the orthogonal loss.
    outs = torch.mean(sims)

    if return_outputs:
        outs = (outs, sims)

    return outs


def triplet_constrastive_loss(args, model, inputs, return_outputs=False):
    gate_logits = None
    loss = 0

    if args.gist_router_aux_loss_coef > 0:
        embeddings_query, embeddings_pos, embeddings_neg, gate_logits = get_embeddings(model, inputs, return_weights=True)

        assert gate_logits is not None, "gate_logits should not be None if args.gist_router_aux_loss_coef > 0"
        aux_loss = compute_load_balancing_loss(gate_logits, top_k=args.gist_topk)

        loss += aux_loss * args.gist_router_aux_loss_coef
    else:
        embeddings_query, embeddings_pos, embeddings_neg = get_embeddings(model, inputs, return_weights=False)

    # Compute the triplet loss for the hard negatives.
    triplet_loss = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1.0 - nn.functional.cosine_similarity(x, y),
        margin=args.gist_tl_margin,
        reduction='mean',
    )
    loss += triplet_loss(embeddings_query, embeddings_pos, embeddings_neg)

    # Compute the contrastive loss excluding the hard negatives. This is done
    # by computing the contrastive loss relative to the in-batch positives and
    # permutation of queries.

    # Remove the hard negatives from the negative embeddings.
    # Set remove_hard_negatives to True to exclude the hard negatives in the contrastive loss computation.
    outs = f_vectorized_in_batch_contrastive_loss(args, embeddings_query, embeddings_pos, embeddings_neg, return_outputs=return_outputs, remove_hard_negatives=False)

    if return_outputs:
        outs = (loss + outs[0], *outs[1:])
    else:
        outs = loss + outs

    return outs


def in_batch_contrastive_loss(args, model, inputs, return_outputs=False, mode="orig"):
    gate_logits = None
    loss = 0

    if args.gist_router_aux_loss_coef > 0:
        embeddings_query, embeddings_pos, embeddings_neg, gate_logits = get_embeddings(model, inputs, return_weights=True)

        assert gate_logits is not None, "gate_logits should not be None if args.gist_router_aux_loss_coef > 0"
        aux_loss = compute_load_balancing_loss(gate_logits, top_k=args.gist_topk)

        loss += aux_loss * args.gist_router_aux_loss_coef
    else:
        embeddings_query, embeddings_pos, embeddings_neg = get_embeddings(model, inputs, return_weights=False)

    if mode == "orig":
        outs = f_in_batch_contrastive_loss(args, embeddings_query, embeddings_pos, embeddings_neg, return_outputs=return_outputs)
    elif mode == "vectorized":
        outs = f_vectorized_in_batch_contrastive_loss(args, embeddings_query, embeddings_pos, embeddings_neg, return_outputs=return_outputs)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if return_outputs:
        outs = (outs[0] + loss, *outs[1:])
    else:
        outs = outs + loss

    return outs


def bidirectional_in_batch_contrastive_loss(args, model, inputs, return_outputs=False):
    embeddings_query, embeddings_pos, embeddings_neg = get_embeddings(model, inputs)
    pass


## Function specifications for each loss function.

def f_orig_in_batch_contrastive_loss(cl_temperature, embeddings_query, embeddings_pos, embeddings_neg, return_outputs=False):

    num = len(embeddings_query)
    all_scores = None

    similarity_fct = nn.CosineSimilarity(dim=-1)
    for i in range(0, num):
        anchor_emb = embeddings_query[i].unsqueeze(0)
        pos_emb = embeddings_pos[i].unsqueeze(0)
        cur_score = similarity_fct(anchor_emb, pos_emb) / cl_temperature

        for j in range(0, num):
            one_neg_emb = embeddings_neg[j].unsqueeze(0)
            one_neg_score = similarity_fct(anchor_emb, one_neg_emb) / cl_temperature
            cur_score = torch.cat([cur_score, one_neg_score], dim=-1)
        if all_scores is None:
            all_scores = cur_score.unsqueeze(0)
        else:
            all_scores = torch.cat([all_scores, cur_score.unsqueeze(0)], dim=0)

    labels = torch.zeros(all_scores.size(0)).long().to(embeddings_query.device)
    loss = nn.CrossEntropyLoss()(all_scores, labels)

    all_another_scores = None
    for i in range(0, num):
        anchor_emb = embeddings_pos[i].unsqueeze(0)
        pos_emb = embeddings_query[i].unsqueeze(0)
        cur_score = similarity_fct(anchor_emb, pos_emb) / cl_temperature

        for j in range(0, num):
            if i == j:
                continue
            one_neg_emb = embeddings_query[j].unsqueeze(0)
            one_neg_score = similarity_fct(anchor_emb, one_neg_emb) / cl_temperature
            cur_score = torch.cat([cur_score, one_neg_score], dim=-1)
        if all_another_scores is None:
            all_another_scores = cur_score.unsqueeze(0)
        else:
            all_another_scores = torch.cat([all_another_scores, cur_score.unsqueeze(0)], dim=0)
    labels_another = torch.zeros(all_another_scores.size(0)).long().to(embeddings_query.device)
    loss += nn.CrossEntropyLoss()(all_another_scores, labels_another)

    if return_outputs:
        return loss, all_scores
    return loss


def f_vectorized_in_batch_contrastive_loss(args, embeddings_query, embeddings_pos, embeddings_neg, return_outputs=False, remove_hard_negatives=False):
    similarity_fct = nn.CosineSimilarity(dim=-1)

    all_scores = vectorized_compute_scores(
        args=args,
        anchor_embeddings=embeddings_query,
        pos_embeddings=embeddings_pos,
        neg_embeddings=embeddings_neg,
        similarity_fct=similarity_fct,
        index=remove_hard_negatives,
    )

    labels = torch.zeros(all_scores.size(0)).long().to(embeddings_query.device)
    loss = nn.CrossEntropyLoss()(all_scores, labels)

    all_another_scores = vectorized_compute_scores(
        args=args,
        anchor_embeddings=embeddings_pos,
        pos_embeddings=embeddings_query,
        neg_embeddings=embeddings_query,
        similarity_fct=similarity_fct,
        index=True,
    )

    labels_another = torch.zeros(all_another_scores.size(0)).long().to(embeddings_query.device)
    loss += nn.CrossEntropyLoss()(all_another_scores, labels_another)

    if return_outputs:
        return loss, all_scores

    return loss


def f_in_batch_contrastive_loss(args, embeddings_query, embeddings_pos, embeddings_neg, return_outputs=False):
    num = len(embeddings_query)
    all_scores = None
    similarity_fct = nn.CosineSimilarity(dim=-1)

    for i in range(0, num):
        cur_score = compute_scores(
            args, num,
            anchor_emb=embeddings_query[i].unsqueeze(0),
            pos_emb=embeddings_pos[i].unsqueeze(0),
            embeddings_neg=embeddings_neg,
            similarity_fct=similarity_fct,
        )
        if all_scores is None:
            all_scores = cur_score.unsqueeze(0)
        else:
            all_scores = torch.cat([all_scores, cur_score.unsqueeze(0)], dim=0)

    labels = torch.zeros(all_scores.size(0)).long().to(embeddings_query.device)
    loss = nn.CrossEntropyLoss()(all_scores, labels)

    all_another_scores = None
    for i in range(0, num):
        cur_score = compute_scores(
            args, num,
            anchor_emb=embeddings_pos[i].unsqueeze(0),
            pos_emb=embeddings_query[i].unsqueeze(0),
            embeddings_neg=embeddings_query,  # Use the queries of other samples in the batch as negatives.
            similarity_fct=similarity_fct,
            index=i,
        )
        if all_another_scores is None:
            all_another_scores = cur_score.unsqueeze(0)
        else:
            all_another_scores = torch.cat([all_another_scores, cur_score.unsqueeze(0)], dim=0)

    labels_another = torch.zeros(all_another_scores.size(0)).long().to(embeddings_query.device)
    loss += nn.CrossEntropyLoss()(all_another_scores, labels_another)

    if return_outputs:
        return loss, all_scores

    return loss


# def f_vectorized_bidirectional_in_batch_contrastive_loss(args, embeddings_query, embeddings_pos, embeddings_neg, return_outputs=False):
#     from torch import nn
#     similarity_fct = nn.CosineSimilarity(dim=-1)

#     all_scores = vectorized_compute_scores(
#         args=args,
#         anchor_embeddings=embeddings_query,
#         pos_embeddings=embeddings_pos,
#         neg_embeddings=embeddings_neg,
#         similarity_fct=similarity_fct,
#         index=None,
#     )

#     labels = torch.zeros(all_scores.size(0)).long().to(embeddings_query.device)
#     loss = nn.CrossEntropyLoss()(all_scores, labels)

#     all_another_scores = vectorized_compute_scores(
#         args=args,
#         anchor_embeddings=embeddings_pos,
#         pos_embeddings=embeddings_query,
#         neg_embeddings=embeddings_query,
#         similarity_fct=similarity_fct,
#         index=True,
#     )

#     labels_another = torch.zeros(all_another_scores.size(0)).long().to(embeddings_query.device)
#     loss += nn.CrossEntropyLoss()(all_another_scores, labels_another)

#     if return_outputs:
#         return loss, all_scores

#     return loss




def load_balancing_loss_func(gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.

    Note: https://github.com/huggingface/transformers/issues/28255#issuecomment-1874241942
    """
    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1) # [batch_size X sequence_length, top_k]

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts) # [batch_size X sequence_length, top_k, num_experts]

    tokens_per_expert = torch.mean(expert_mask.float(), dim=0) # [top_k, num_experts]

    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0) # [num_experts]
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0)) # / top_k

    # return (routing_weights, selected_experts, expert_mask, tokens_per_expert, router_prob_per_expert, overall_loss, overall_loss * num_experts)

    return overall_loss * num_experts


def hierarchical_contrastive_loss(args, model, inputs, return_outputs=False):
    """We extract subembeddings from the sentence embeddings and compute the
    contrastive loss on the subembeddings.

    The subembeddings are extracted by randomly selecting a subsequence of the
    sentence embeddings. We use mutually exclusive elements in the subsequence
    to avoid overlapping subembeddings.

    The intuition is that we want to encourage the model to learn the relationships
    across dimensions of the sentence embeddings. Higher dimensional sentence embeddings
    tend to have more information, but they are also more difficult to learn. By regularizing
    the model to learn the relationships across dimensions, we can align contrast across the
    full dimensionality of the sentence embeddings.
    """

    # Generate the subembeddings mask.
    from random import sample

    assert not return_outputs, "return_outputs is not supported for hierarchical contrastive loss"

    embeddings_query, embeddings_pos, embeddings_neg = get_embeddings(model, inputs, return_weights=False)
    device = embeddings_query.device

    # Compute the contrastive loss on the full embeddings.
    loss = f_vectorized_in_batch_contrastive_loss(args, embeddings_query, embeddings_pos, embeddings_neg, return_outputs=return_outputs)

    # Iterate args.gist_hcl_num_subembeddings times to compute the subembeddings loss.
    mask_idx = list(range(embeddings_query.size(-1)))
    num_samples = embeddings_query.size(-1) // args.gist_hcl_num_subembeddings

    for _ in range(args.gist_hcl_num_subembeddings):
        # Sample the indices of the subembeddings.

        # If the number of remaining indices is greater than 1.2 times the number of samples,
        # then we can proceed sampling the indices without replacement.
        if len(mask_idx) >= (1.2 * num_samples):
            subembeddings_idx = sample(mask_idx, num_samples)
            mask_idx = list(set(mask_idx).difference(subembeddings_idx))
        else:
            # Otherwise, we use the remaining indices.
            subembeddings_idx = mask_idx

        subembeddings_idx = torch.tensor(subembeddings_idx).long().to(device)

        q_emb = embeddings_query.index_select(dim=-1, index=subembeddings_idx)
        p_emb = embeddings_pos.index_select(dim=-1, index=subembeddings_idx)
        n_emb = embeddings_neg.index_select(dim=-1, index=subembeddings_idx)

        loss += f_vectorized_in_batch_contrastive_loss(args, q_emb, p_emb, n_emb, return_outputs=return_outputs)

    return loss



def bm25_loss(args, model, inputs, return_outputs=False):
    """This loss function is used to replicate the BM25 loss function in the
    context of transformer-based models. The BM25 loss function is defined as:
    BM25(q, d) = sum_{i=1}^{n} log(1 + (k_1 + 1) * tf_{i, q} / (k_1 * (1 - b + b * L_d / L_avg) + tf_{i, q})) * idf_i
    where
    - n: number of query terms
    - tf_{i, q}: term frequency of the i-th query term in the document
    - L_d: document length
    - L_avg: average document length
    - k_1, b: hyperparameters
    - idf_i: inverse document frequency of the i-th query term
    """

    def bm25(query, samples):
        pass

    inputs



def guided_in_batch_contrastive_loss(args, guide, model, inputs, return_outputs=False):
    loss = 0

    similarity_fct = nn.CosineSimilarity(dim=-1)
    embeddings_query, embeddings_pos, embeddings_neg = get_embeddings(model, inputs, return_weights=False)

    # Compute the model's similarities
    qp_sim = similarity_fct(embeddings_query.unsqueeze(1), embeddings_pos.unsqueeze(0))
    qn_sim = similarity_fct(embeddings_query.unsqueeze(1), embeddings_neg.unsqueeze(0))
    qq_sim = similarity_fct(embeddings_query.unsqueeze(1), embeddings_query.unsqueeze(0))
    pp_sim = similarity_fct(embeddings_pos.unsqueeze(1), embeddings_pos.unsqueeze(0))

    if guide is None:
        # Adopt the contrastive loss without the guide based on the partition function
        # used in: https://arxiv.org/pdf/2308.03281.pdf
        # Towards General Text Embeddings with Multi-stage Contrastive Learning
        # This loss function is a bit different compared with the bidirectional contrastive loss
        # used in the INSTRUCTOR paper (https://arxiv.org/pdf/2212.09741.pdf), particularly
        # because we don't split the partition function for the query and the document as
        # two separate components of the loss.

        # Mask the diagonal elements for i==j query and document pairs.
        qq_sim.fill_diagonal_(-torch.inf)
        pp_sim.fill_diagonal_(-torch.inf)
    else:
        # Compute the guide's similarities
        guide_embeddings_query, guide_embeddings_pos, guide_embeddings_neg = get_guide_embeddings(guide, inputs)

        # This contains the cosine similarities of the query with respect to hard negatives
        guided_qp_sim = similarity_fct(guide_embeddings_query.unsqueeze(1), guide_embeddings_pos.unsqueeze(0))
        guided_qn_sim = similarity_fct(guide_embeddings_query.unsqueeze(1), guide_embeddings_neg.unsqueeze(0))
        guided_qq_sim = similarity_fct(guide_embeddings_query.unsqueeze(1), guide_embeddings_query.unsqueeze(0))
        guided_pp_sim = similarity_fct(guide_embeddings_pos.unsqueeze(1), guide_embeddings_pos.unsqueeze(0))

        guided_sim = guided_qp_sim.diagonal().view(-1, 1)

        # Find which samples cannot be used as negatives because they are
        # more similar to the query than the assigned positive as deemed by the guide model.
        # For this samples, we mask them with -inf to basically ignore their contribution to
        # the loss.
        qp_mask = guided_qp_sim > guided_sim
        qn_mask = guided_qn_sim > guided_sim
        qq_mask = guided_qq_sim > guided_sim  # This should take care of masking the diagonal.
        pp_mask = guided_pp_sim > guided_sim  # This should take care of masking the diagonal.

        if args.gist_loss_type.startswith("guided-triplet") and args.gist_tl_margin > 0:
            # If we use triplet loss, we go here.
            # We use all the q* matrices as basis for computing
            # the triplet loss.
            qp_tr_mask = guided_qp_sim >= guided_sim

            # From the qp, qn, and qq masks, we find values greater than or equal to guided_sim
            # as the pool of candidate positives. Less than the guided_sim are candidate negatives.

            # For a given query, we find the sample with lowest similarity score among the candidate positives
            # as the triplet positive. We also choose the sample with the highest similarity score among
            # the candidate negatives as the triplet positive.

            # These are "positive" pairs.
            tr_mask = torch.cat([qp_tr_mask, qn_mask, qq_mask], dim=1)
            q = torch.cat([qp_sim, qn_sim, qq_sim], dim=1)

            if args.gist_loss_type == "guided-triplet":
                # Hard
                qp_dist = (q * tr_mask).argmin(dim=1)
                qn_dist = (q * (~tr_mask)).argmax(dim=1)

                # Subtract from 1 to get the cosine distance.
                _p = 1 - q[torch.arange(q.size(0)).long().to(q.device), qp_dist]
                _n = 1 - q[torch.arange(q.size(0)).long().to(q.device), qn_dist]
            elif args.gist_loss_type == "guided-triplet-soft":
                # Soft
                qp_dist = (q * tr_mask)
                qn_dist = (q * (~tr_mask))

                # Note that since qq_mask contains the self similarity of q-q.
                # We should adjust for that below by subtracting 1 in the similarity
                # score and the total components in the denominator.
                # We only do this adjustment to the qp_dist.

                # Convert to cosine distance
                qp_dist[tr_mask] = 1 - qp_dist[tr_mask]
                qn_dist[~tr_mask] = 1 - qn_dist[~tr_mask]

                # Get the mean accounting for q-q in the denominator (-1)
                p_den = tr_mask.sum(dim=-1) - 1
                n_den = (~tr_mask).sum(dim=-1)

                # We exclude in the loss computation samples without negatives
                # or positives.
                sample_mask = (p_den != 0) & (n_den != 0)

                p_den = p_den[sample_mask]
                n_den = n_den[sample_mask]

                _p = qp_dist.sum(dim=-1)[sample_mask] / p_den
                _n = qn_dist.sum(dim=-1)[sample_mask] / n_den
            else:
                raise ValueError(f"Unsupported loss type: {args.gist_loss_type}")

            # Hard triplet margin loss
            tml = (_p - _n + args.gist_tl_margin)
            tml[tml < 0] = 0

            loss += torch.mean(tml)

        if args.gist_negative_mode == "hard":
            # Find the hard negatives defined by examples that the model
            # finds more similar to the query than the assigned positive, but
            # the guide model finds them as negatives.

            model_sim = qp_sim.diagonal().view(-1, 1)

            # We find samples the model already finds as negatives.
            # This means the similarity of the query with respect to the positive
            # is greater than the similarity of the query with respect to the other potential negatives.
            model_qp_mask = qp_sim < model_sim
            model_qn_mask = qn_sim < model_sim
            model_qq_mask = qq_sim < model_sim
            model_pp_mask = pp_sim < model_sim

            qp_mask = qp_mask | model_qp_mask
            qn_mask = qn_mask | model_qn_mask
            qq_mask = qq_mask | model_qq_mask
            pp_mask = pp_mask | model_pp_mask

            # # TODO: We need to handle cases where there are no hard negatives left.
            # mask = torch.cat([qp_mask, qn_mask, qq_mask, pp_mask], dim=1)
            # num_negatives = (~mask).sum(dim=1)

            # # We find samples without hard negatives left.
            # # This means the similarity of the query with respect to the positive
            # # is greater than the similarity of the query with respect to the other potential negatives.

        qp_sim[qp_mask] = -torch.inf
        qn_sim[qn_mask] = -torch.inf
        qq_sim[qq_mask] = -torch.inf
        pp_sim[pp_mask] = -torch.inf

    scores = torch.cat([qp_sim, qn_sim, qq_sim, pp_sim], dim=1) / args.gist_cl_temperature
    labels = torch.arange(scores.size(0)).long().to(embeddings_query.device)
    loss += nn.CrossEntropyLoss()(scores, labels)

    if return_outputs:
        # Note that we only return the contrastive loss scores here.
        outs = (loss, scores)
    else:
        outs = loss

    return outs
