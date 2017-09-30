% Case1 
% Interclass separation is high
% Intraclass separation is low
% Fisher score should be high

data1 = [1, 2, 3, 4, 5];
data2 = [21, 22, 23, 24, 25];

hAxes1 = axes('NextPlot','add',...           %# Add subsequent plots to the axes,
             'DataAspectRatio',[1 1 1],...  %#   match the scaling of each axis,
             'XLim',[0 30],...               %#   set the x axis limit,
             'YLim',[0 eps],...             %#   set the y axis limit (tiny!),
             'Color','none');               %#   and don't use a background color
plot(data1,0,'r*','MarkerSize',10);  %# Plot data set 1
plot(data2,0,'b.','MarkerSize',10);  %# Plot data set 2

% Calculate Fisher score
mean_class1 = mean(data1);
mean_class2 = mean(data2);
std_class1 = std(data1);
std_class2 = std(data2);
mean_class1_2 = mean(horzcat(data1, data2)); 
avg_interclass_separ = 1/2 * ((mean_class1_2 - mean_class1)^2 + (mean_class1_2 - mean_class2)^2);
avg_intraclass_separ = 1/2 * (std_class1^2 + std_class2^2);
Fisher_score1 = avg_interclass_separ / avg_intraclass_separ;
fprintf('Fisher_score1 is %f\n', Fisher_score1);


% Case2
% Interclass separation is low
% Intraclass separation is high
% Fisher score should be low

data1 = [1, 4, 7, 10, 13];
data2 = [16, 19, 22, 25, 28];

figure;
hAxes2 = axes('NextPlot','add',...           %# Add subsequent plots to the axes,
             'DataAspectRatio',[1 1 1],...  %#   match the scaling of each axis,
             'XLim',[0 30],...               %#   set the x axis limit,
             'YLim',[0 eps],...             %#   set the y axis limit (tiny!),
             'Color','none');               %#   and don't use a background color
plot(data1,0,'r*','MarkerSize',10);  %# Plot data set 1
plot(data2,0,'b.','MarkerSize',10);  %# Plot data set 2

% Calculate Fisher score
mean_class1 = mean(data1);
mean_class2 = mean(data2);
std_class1 = std(data1);
std_class2 = std(data2);
mean_class1_2 = mean(horzcat(data1, data2)); 
avg_interclass_separ = 1/2 * ((mean_class1_2 - mean_class1)^2 + (mean_class1_2 - mean_class2)^2);
avg_intraclass_separ = 1/2 * (std_class1^2 + std_class2^2);
Fisher_score2 = avg_interclass_separ / avg_intraclass_separ;
fprintf('Fisher_score2 is %f\n', Fisher_score2);