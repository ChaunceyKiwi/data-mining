data = randi(100, 10, 1);

% Min-max scaling
max_val = max(data);
min_val = min(data);
data_scaled = (data - min_val) / (max_val - min_val);


% Z-score
mean_val = mean(data);
std_val = std(data);
data_z_score = (data - mean_val) / std_val;

% Percentile rank
N = size(data, 1);
data_percentile = zeros(N, 1);
for i = 1 : N
    data_percentile(i) = (sum(data < data(i)) + 0.5 * sum(data == data(i))) / N;
end

% display result
subplot(2,2,1), histogram(data, 10); title('Original Data');
subplot(2,2,2), histogram(data_scaled, 10); title('Min-max scaling');
subplot(2,2,3), histogram(data_z_score, 10); title('Z-score');
subplot(2,2,4), histogram(data_percentile, 10); title('Percentile rank');