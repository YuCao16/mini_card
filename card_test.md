# Structure of test

- `compute_prediction_SE( config, dataset_object, y_batch, generated_y, return_pred_mean=False ) : function`

- `zzcompute_true_coverage_by_gen_QI( config, dataset_object, all_true_y, all_generated_y, verbose=True ) : functionzz`

- `zcompute_PICP(config, y_true, all_gen_y, return_CI=False) : functionz`

- `store_gen_y_at_step_t(config, current_batch_size, idx, y_tile_seq) : function`

- `store_y_se_at_step_t(config, idx, dataset_object, y_batch, gen_y) : function`

- `set_NLL_global_precision(test_var=True) : function`

- `compute_batch_NLL(config, dataset_object, y_batch, generated_y) : function`

- `store_nll_at_step_t(config, idx, dataset_object, y_batch, gen_y) : function`
