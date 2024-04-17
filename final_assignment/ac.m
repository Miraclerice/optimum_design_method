function[dist_best,path_best,per_iter_best,per_iter_avg]=AC_GA(map,start_pos_ind, goal_pos_ind)
while NC<=NC_max
    for ant=1:m
        current_position=s;
        path=s;
        PL_NC_ant=0;%%长度初始化
        Tabu=ones(1,n*n); %%%%禁忌表，排除已经走过的位置
        Tabu(s)=0;%%排除已经走过的初始点
        D_D=D_move;
        D_work=D_D(current_position,:);%%%把当前点可以前往的下一个节点的信息传送给D_work
        nonzeros_D_work=find(D_work);%%%找到不为0 的元素的位置
        for i1=1:length(nonzeros_D_work)
            if Tabu(D_work(i1))==0
                D_work(nonzeros_D_work(i1))=[];
                D_work=[D_work,zeros(1,8-length(D_work))];
            end
        end
        len_D_work=length(find(D_work));
        while current_position~=position_e&&len_D_work>=1
            p=zeros(1,len_D_work);
            for j1=1:len_D_work
                [r1,c1]=position2rc(D_work(j1));
                p(j1)=(Tau(r1,c1)^Alpha)*(Eta(r1,c1)^Beta);
            end
            p=p/sum(p);
            pcum=cumsum(p);
            select=find(pcum>=rand);
            to_visit=D_work(select(1));
            path=[path,to_visit];%%%路径累加
            dis=distance(current_position,to_visit);
            PL_NC_ant=PL_NC_ant+dis;
            current_position=to_visit;%%%当前点设为前往点
            D_work=D_D(current_position,:);
            Tabu(current_position)=0;
            for kk=1:400
                if Tabu(kk)==0
                    for i3=1:8
                        if D_work(i3)==kk
                            D_work(i3)=[];
                            D_work=[D_work,zeros(1,8-length(D_work))];
                        end
                    end
                end
            end
            len_D_work=length(find(D_work));
        end
        routes{NC,ant}=path;
        if path(end)==position_e
            PL(NC,ant)=PL_NC_ant;
            if PL_NC_ant<min_PL_NC_ant
                min_NC=NC;min_ant=ant;min_PL_NC_ant=PL_NC_ant;
            end
        else
            PL(NC,ant)=0;
        end
    end
    delta_Tau=zeros(n,n);
    for j3=1:m
        if PL(NC,ant)
            rout=routes{NC,ant};
            tiaoshu=length(rout)-1;
            value_PL=PL(NC,ant);
            for u=1:tiaoshu
                [r3,c3]=position2rc(rout(u+1));
                delta_Tau(r3,c3)=delta_Tau(r3,c3)+Q/value_PL;
            end
        end
    end
    Tau=(1-Rho).*Tau+delta_Tau;
    NC=NC+1;
end
path_value=calculation_path_value(new_population);
[sort1,index]=sort(path_value);
new_population=new_population(index);
path_value=calculation_path_value(new_population);
smooth_value=calculation_smooth_value(new_population);
fit_value=(weight_length./path_value)+(weight_smooth./smooth_value);
mean_path_value=zeros(1,max_generation);
min_path_value=zeros(1,max_generation);
for i=1:max_generation
    new_population_1=selection(new_population,fit_value); %选择
    new_population_1=crossover(new_population_1,p_crossover); %交叉
    new_population_1=mutation(new_population_1,p_mutation,G,r); %突变
    new_population_1=GenerateSmoothPath(new_population_1,G);
    new_population=new_population_1;
    path_value=calculation_path_value(new_population);
    smooth_value=calculation_smooth_value(new_population);
    fit_value=(weight_length./path_value)+(weight_smooth./smooth_value); %适应度计算
    mean_path_value(i)=mean(path_value);
    [~,ma]=max(fit_value);
    min_path_value(i)=path_value(ma);
    min_path{i}=new_population(ma);
end