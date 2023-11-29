import src.movement_rates as mr



def analysis_rates(raw_rates_input_file, results_output_path, raw_rates_output_filename,
                   mean_normalized_output_filename, rate_parameters_output_filename, permutest_output_filename,
                   onset_column, offset_column, participant_column, touch_order_column,
                   analysis_parameter_dict, permutation_dict, conditions_dict,
                   dependent_vars, independent_vars, baseline_condition_name):
    """
    Movement Rate Analysis
    - Step 1: Compute rates for individual participants
    - Step 2: Normalize rates to the mean baseline
    - Step 3: Get metrics from the rates
    - Step 4: Compare metrics statistically
    - Step 5: Compare rates with a cluster based permutation test

    This function calls all the above steps with parameters defined in the dictionaries (analysis_parameter_dict,
    permutation_dict, conditions_dict)

    raw_rates_input_file: path to the preprocessed data file (each touch is one row)
    results_ouput_path: path to the results folder
    raw_rates_output_filename: filename under which results from Step 1 are saved
    mean_normalized_output_filename: filename under which results from Step 2 are saved
    rate_parameters_output_filename: filename under which results from Step 3 are saved
    permutest_output_filename: filename under which results from Step 5 are saved

    Results from Step 4 are printed into the console.

    onset_column: the column name from raw_input_file that contains touch onsets relative to the change
    offset_column: the column name from raw_input_file that contains touch offsets relative to the change
    participant_column: the column name from raw_input_file that contains the participant id
    touch_order_column: the column name from raw_input_file that contains the order in which the taps were made

    analysis_parameter_dict: a dictionary defining the movement rate analysis. Must have the following keys:
    - window_start: start of the time scale (-700 in Kuper & Rolfs 2024)
    - window_end: end of the time scale (1500 in Kuper & Rolfs 2024)
    - alpha: the width of the moving causal kernel used to compute movement rates (1/50 in Kuper & Rolfs 2024)
    - search_start: start of the time window in which we look for movement parameters (0 in Kuper & Rolfs 2024)
    - search_end: end of the time window in which we look for movement parameters (700 in Kuper & Rolfs 2024)

    permutation_dict: a dictionary defining the permutation test. Must have the following keys:
    - baseline_name: name of the condition we want to compare to ('flash- jump-')
    - t_value: cutoff value for significant t-values (2.093)
    - n_permutations: the number of permutations to perform (1000)
    - percentile_cutoff: the upper percentile to consider (0.05)
    """

    # load the table with touch responses, save data file with movement rates
    raw_rates_output_path = mr.get_movement_rates_by_participant(raw_rates_input_file, results_output_path, raw_rates_output_filename,
                                                                 onset_column, offset_column, participant_column, touch_order_column,
                                                                 analysis_parameter_dict, conditions_dict)

    # load raw movement rates, save rates normalized to mean baseline and save
    normalized_rates_output_path = mr.mean_normalize_rates(raw_rates_output_path, results_output_path, mean_normalized_output_filename,
                                                           baseline_condition_name)

    # save parameters from the movement rates
    rate_parameters_output_path = mr.get_movement_rate_parameters(normalized_rates_output_path, results_output_path, rate_parameters_output_filename,
                                                                  analysis_parameter_dict)

    # run ANOVA on parameters
    mr.run_anovas(rate_parameters_output_path, dependent_vars, independent_vars, 'participant')

    # perform a cluster-based permutation test to identify significant differences
    cluster_output_path = mr.perform_cluster_based_permutation(normalized_rates_output_path, results_output_path, permutest_output_filename,
                                                               permutation_dict['baseline_name'], permutation_dict['t_value'],
                                                               permutation_dict['n_permutations'], permutation_dict['percentile_cutoff'])


