python plot_metric_cdf.py \
    --scheme ./results/mcmc_test_results.json \
             ./results/greedy_test_results.json \
             ./results/graph_cut_test_results.json \
             ./results/spectral_test_results.json \
    --scheme_names MCMC greedy graph_cut spectral \
    --output_dir outputs

python plot_metric_cdf.py \
  --metric success_6A \
  --scheme MCMC=./results/mcmc_test_results.json \
  --scheme greedy=./results/greedy_test_results.json \
  --scheme graph_cut=./results/graph_cut_test_results.json  \
  --scheme spectral=./results/spectral_test_results.json \
  --output ./pic/success_6A_cdf.png