function X = labelClustersManually(fileName)

%% Display file path
filePath = strcat('../data/rawClusters/clusters_', fileName, '.csv');
disp(strcat('Opening file in: ', filePath));

%% Variables configurables
samplingFreq = 30;
t = 0;

%% Abrir el archivo csv
clusterBag = csvread(filePath, 1, 1);

[nRows, ~] = size(clusterBag);
M = [];

for i = 1:nRows
    % show points
    n = rowSizeUntilNan(clusterBag(i,:));
    nClusters = n/3;
    clusters = clusterBag(i,1:3:3*nClusters);
    x = clusterBag(i,2:3:3*nClusters);
    y = clusterBag(i,3:3:3*nClusters);
    plot(0,0,'*',x,y,'o')
    %text(x+.2,y+.2,strtrim(cellstr(num2str(clusters'))'));
    text(x+.2,y+.2,strtrim(cellstr(num2str((1:nClusters)'))'));
    xlim([-15 15]);
    ylim([-15 15]);

    % ask the user
    cluster = input('Cluster num? ');
    if cluster
        x_ = x(cluster);
        y_ = y(cluster);
        % save
        M(end + 1,:) = [t, clusters(cluster), x_, y_];
    end
    % update time
    t = t + 1/samplingFreq;
end

csvPath = strcat('../data/labeledClusters/end_clusters_', fileName, '.csv');
disp(strcat('Saving to: ', csvPath));
csvwrite(csvPath, M);
end