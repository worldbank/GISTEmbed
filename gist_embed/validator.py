def validate_script_id(gist_args, model_args, training_args):
    script_id = gist_args.gist_script_id

    if script_id is None:
        return

    # Refer to EXPERIMENTS.md for the list of valid script IDs.
    pool = None
    normed = None
    out_dim = None
    inseq = None
    negative = None

    if script_id.count("-") == 11:
        XX, YYY, ZZ, a, b, c, d, e, pool, normed, out_dim, inseq = script_id.split("-")
    elif script_id.count("-") == 12 and '-hard_' in script_id:
        XX, YYY, ZZ, a, b, c, d, e, pool, normed, out_dim, inseq, negative = script_id.split("-")
    elif script_id.count("-") == 12:
        XX, YYY, ZZ, a, b, c, d, e, pool, normed, out_dim, inseq, grad_acc_steps = script_id.split("-")
        # Cast this to int so we know that this is very corresponds to the grad_acc_steps.
        # Add better handling later.
        grad_acc_steps = int(grad_acc_steps)
        assert training_args.gradient_accumulation_steps == grad_acc_steps
    else:
        XX, YYY, ZZ, a, b, c, d, e = script_id.split("-")

    if negative is not None:
        assert gist_args.gist_negative_mode == negative

    if pool is not None:
        assert gist_args.gist_auto_model_pooling == pool

    if normed is not None:
        if normed == "normed":
            assert gist_args.gist_normalize
        elif normed == "unnormed":
            assert not gist_args.gist_normalize
        else:
            raise ValueError(f"Unregistered normed: {normed}")

    if out_dim is not None:
        assert gist_args.gist_output_dim == int(out_dim)

    if inseq is not None:
        assert gist_args.max_source_length == int(inseq)

    if XX == "00":
        assert model_args.model_name_or_path == "BAAI/bge-base-en-v1.5"
    elif XX == "01":
        assert model_args.model_name_or_path == "BAAI/bge-small-en-v1.5"
    elif XX == "02":
        assert model_args.model_name_or_path == "BAAI/bge-large-en-v1.5"
    elif XX == "10":
        assert model_args.model_name_or_path == "infgrad/stella-base-en-v2"
    elif XX == "20":
        assert model_args.model_name_or_path == "avsolatorio/BAAI_bge-base-en-v1.5_1024"
        assert gist_args.max_source_length == 1024
    elif XX == "30":
        assert model_args.model_name_or_path == "sentence-transformers/all-MiniLM-L6-v2"
        assert pool == "mean", "Only mean pooling is supported for sentence-transformers/all-MiniLM-L6-v2. Explicitly update the validator if you want to use another pooling strategy."
    elif XX == "40":
        assert model_args.model_name_or_path == "TaylorAI/gte-tiny"
        assert pool == "mean", "Only mean pooling is supported for TaylorAI/gte-tiny. Explicitly update the validator if you want to use another pooling strategy."
    elif XX == "50":
        assert model_args.model_name_or_path == "WhereIsAI/UAE-Large-V1"
        assert pool == "cls", "Only cls pooling is supported for WhereIsAI/UAE-Large-V1. Explicitly update the validator if you want to use another pooling strategy."
    else:
        raise ValueError(f"Unregistered XX: {XX}")

    if YYY[0] == "0":
        assert gist_args.gist_medi_data_name == "avsolatorio/medi-data"
    elif YYY[0] == "1":
        assert gist_args.gist_medi_data_name == "avsolatorio/medi-data-mteb_avs_triplets"
    elif YYY[0] == "2":
        assert gist_args.gist_medi_data_name == "avsolatorio/medi-data-sorted_WhereIsAI_UAE-Large-V1"
    elif YYY[0] == "3":
        assert gist_args.gist_medi_data_name == "avsolatorio/medi-data-sorted_WhereIsAI_UAE-Large-V1-mteb_avs_triplets"
    elif YYY[0] == "4":
        assert gist_args.gist_medi_data_name == "avsolatorio/medi-data-mteb-wb-prwp-covid_avs_triplets"
    elif YYY[0] == "5":
        assert gist_args.gist_medi_data_name == "avsolatorio/medi-data-mteb-wb-prwp-covid-sent-avs_triplets"
    elif YYY[0] == "6":
        assert gist_args.gist_medi_data_name == "avsolatorio/medi-data-mteb-covid-bing-query-gpt4-avs_triplets"
    else:
        raise ValueError(f"Unregistered YYY: {YYY}")


    if ZZ == "00":
        assert gist_args.gist_loss_type == "contrastive"
    elif ZZ == "10":
        assert gist_args.gist_loss_type == "guided" and gist_args.gist_guide_model_name_or_path is None
    elif ZZ[1] == "1":
        assert gist_args.gist_guide_model_name_or_path == "WhereIsAI/UAE-Large-V1"
        if ZZ[0] == "1":
            assert gist_args.gist_loss_type == "guided"
        elif ZZ[0] == "2":
            assert gist_args.gist_loss_type == "guided-triplet" and gist_args.gist_tl_margin > 0
        elif ZZ[0] == "3":
            assert gist_args.gist_loss_type == "guided-triplet-soft" and gist_args.gist_tl_margin > 0
        else:
            raise ValueError(f"{ZZ} != {gist_args.gist_loss_type} and {gist_args.gist_guide_model_name_or_path}")
    elif ZZ[1] == "2":
        assert gist_args.gist_guide_model_name_or_path == "BAAI/bge-large-en-v1.5"
        if ZZ[0] == "1":
            assert gist_args.gist_loss_type == "guided"
        elif ZZ[0] == "2":
            assert gist_args.gist_loss_type == "guided-triplet" and gist_args.gist_tl_margin > 0
        elif ZZ[0] == "3":
            assert gist_args.gist_loss_type == "guided-triplet-soft" and gist_args.gist_tl_margin > 0
        else:
            raise ValueError(f"{ZZ} != {gist_args.gist_loss_type} and {gist_args.gist_guide_model_name_or_path}")
    else:
        raise ValueError(f"Unregistered ZZ: {ZZ}")

    if a == "0":
        assert training_args.learning_rate == 2e-5
    elif a == "1":
        assert training_args.learning_rate == 5e-6
    elif a == "2":
        assert training_args.learning_rate == 1e-4
    else:
        raise ValueError(f"Unregistered a: {a}")

    if b == "0":
        assert (training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size) == 4
        assert training_args.num_train_epochs == 10
    elif b == "1":
        assert (training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size) == 8
        assert training_args.num_train_epochs == 20
    elif b == "2":
        assert (training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size) == 16
        assert training_args.num_train_epochs == 40
    elif b == "3":
        assert (training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size) == 32
        assert training_args.num_train_epochs == 80
    elif b == "3.5":
        assert (training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size) == 48
        assert training_args.num_train_epochs == 120
    elif b == "3.75":
        assert (training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size) == 56
        assert training_args.num_train_epochs == 140
    elif b == "4":
        assert (training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size) == 64
        assert training_args.num_train_epochs == 160
    elif b == "5":
        assert (training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size) == 128
        assert training_args.num_train_epochs == 320
    else:
        raise ValueError(f"Unregistered b: {b}")

    if c == "0":
        assert training_args.num_train_epochs == 1
    elif c == "1":
        assert training_args.num_train_epochs == 10
    elif c == "2":
        # We use this since epochs should have been checked
        # above to match the batch size.
        pass
    else:
        raise ValueError(f"Unregistered c: {c}")

    if d == "0":
        assert training_args.warmup_ratio == 0.1
    elif d == "1":
        assert training_args.warmup_ratio == 0.2
    else:
        raise ValueError(f"Unregistered d: {d}")

    if e == "0":
        assert gist_args.gist_cl_temperature == 0.01
    elif e == "1":
        assert gist_args.gist_cl_temperature == 0.1
    else:
        raise ValueError(f"Unregistered e: {e}")
