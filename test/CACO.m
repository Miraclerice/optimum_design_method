[center]=GACO(n,m,m_min,y_max,y_min,C,alpha ,Q,itr_max)
rho = 0.1; % 信息素挥发因子
beta=y_max-(y_max-y_min)*(1/itr_max);%初始值
distance = rand(n,n); % 两点之间的距离
pheromone = ones(n,n); % 信息素矩阵
% 迭代循环
for iter = 1:itr_max
    % 计算距离和信息素的影响因子
    eta = 1./(distance+C);
    tau = pheromone.^alpha;
    path = zeros(m,n);
    path(:,1) = randi(n, m, 1); % 随机选择起始点
    % 开始选择路径
    for i = 1:m
        for j = 2:n
            % 计算概率
            p = tau(path(i,j-1),:) .^ alpha .* eta(path(i,j-1),:) .^ beta;
            p(path(i,:)) = 0; % 排除已经选择的点
            p = p / sum(p);
            path(i,j) = randsrc(1,1,[1:n; p]);
        end
    end
    % 更新信息素矩阵
    delta_pheromone = zeros(n,n);
    for i = 1:m
        for j = 2:n
            delta_pheromone(path(i,j-1),path(i,j)) = delta_pheromone(path(i,j-1),path(i,j)) + Q/distance(path(i,j-1),path(i,j));
         end
    end
    pheromone = (1-rho) * pheromone + delta_pheromone;
    if (m<m_min)
        m=m_min;
        break;
    end
    m=m-m_min*(itr/itr_max);
end
% 找到最佳中心点
center = zeros(1,10);
for i = 1:10
    [~,idx] = max(sum(pheromone,2));
    center(i) = idx;
    pheromone(idx,:) = 0;
    pheromone(:,idx) = 0;
end