clear all;

%load('AdSetting.mat');%To replicate the results with the same basic setting

%% basic setting
%Skip this section if load ``AdSetting.mat''
n = 50;%number of ads 50
m = 5;%number of customer types 5
K = 2;%cadinality constriant 2
lambda = 10;%regularization parameter 10
uV = 5;%upper of distribution of determined utility 5
S = 30;%sample path size 30
tratio = 20;%a ratio of T to n 20
T = n*tratio;%total number of periods 1000
CV=[0.1,1,10,100];%CP parameter
LF=[1.5,1.25,1,0.75,0.5];%LF parameter

r = unifrnd(10,50,n,m);%revenue of ads
rTarget = unifrnd(10,50,n,m);%alternative revenue of ads for giving feasible requirements
V = unifrnd(0,uV,n,m);%utility of customers
eV = exp(V);%parameters of MNL choice model

%customer arrival distribution setting with CP parameter
pSummary = zeros(m,length(CV));
for cvi = 1:length(CV)
    a1 = repmat(CV(cvi),1,m);
    p1 = gamrnd(repmat(a1,1,1),1,1,m);
    p1 = p1./repmat(sum(p1,2),1,m);
    pSummary(:,cvi) = p1';
end

etaTarget = zeros(n,m,length(CV),length(LF));%summary of requirement of clicks
fprintf('Basic settings complete!\n');

%% main simulation
%construct lhs matrix of target optimization
lhsTarget = zeros(n*m+m+n,n*m);
for j = 1:m
    lhsTarget(n*m+j,(j-1)*n+1:(j-1)*n+n) = 1./eV(:,j)'/K+1;
    for i = 1:n
        lhsTarget((j-1)*n+i,(j-1)*n+1:(j-1)*n+n) = ones(1,n);
        lhsTarget((j-1)*n+i,(j-1)*n+i) = lhsTarget((j-1)*n+i,(j-1)*n+i)+1/eV(i,j);
        lhsTarget(n*m+m+i,(j-1)*n+i) = 1;
    end
end

%construct decision set for each offerset of each customer type in Fluid model
deFromBi=zeros(n,1);
for i=1:n
    deFromBi(i)=2^(i-1);
end
sn = 1;
for numkk = 1:K
    sn = sn + nchoosek(n,numkk);
end

CPM=zeros(n+1,m,sn);%whole decision set

%all offersets with cardinality = 1
nchoose1CPM = zeros(n+1,m,nchoosek(n,1));
nchoose1t=1;
for nchoose1i=1:n-1+1
    nchoose1ss=zeros(n,1);
    nchoose1ss(nchoose1i)=1;
    for tyi=1:m
        nchoose1CPM(1,tyi,nchoose1t)=nchoose1ss'*deFromBi;
        nchoose1CPM(2:n+1,tyi,nchoose1t)=nchoose1ss.*eV(:,tyi)/(1+nchoose1ss'*eV(:,tyi));
    end
    nchoose1t=nchoose1t+1;
end

%all offersets with cardinality = 2
nchoose2CPM = zeros(n+1,m,nchoosek(n,2));
nchoose2t=1;
for nchoose2i=1:n-2+1
    for nchoose2j=nchoose2i+1:n-2+2
        nchoose2ss=zeros(n,1);
        nchoose2ss(nchoose2i)=1;
        nchoose2ss(nchoose2j)=1;
        for tyi=1:m
            nchoose2CPM(1,tyi,nchoose2t)=nchoose2ss'*deFromBi;
            nchoose2CPM(2:n+1,tyi,nchoose2t)=nchoose2ss.*eV(:,tyi)/(1+nchoose2ss'*eV(:,tyi));
        end
        nchoose2t=nchoose2t+1;
    end
end

%all offersets with cardinality = 3
% nchoose3CPM = zeros(n+1,m,nchoosek(n,3));
% nchoose3t=1;
% for nchoose3i=1:n-3+1
% for nchoose3j=nchoose3i+1:n-3+2
% for nchoose3k=nchoose3j+1:n-3+3
%     nchoose3ss=zeros(n,1);
%     nchoose3ss(nchoose3i)=1;
% nchoose3ss(nchoose3j)=1;
% nchoose3ss(nchoose3k)=1;
%     for tyi=1:m
% nchoose3CPM(1,tyi,nchoose3t)=nchoose3ss'*deFromBi;
% nchoose3CPM(2:n+1,tyi,nchoose3t)=nchoose3ss.*eV(:,tyi)/(1+nchoose3ss'*eV(:,tyi));
%  end
% nchoose3t=nchoose3t+1;
% end
% end
% end

CPM(:,:,2:nchoosek(n,1)+1)= nchoose1CPM;
CPM(:,:,nchoosek(n,1)+1+1:nchoosek(n,1)+1+nchoosek(n,2))= nchoose2CPM;
%CPM(:,:,nchoosek(n,1)+1+nchoosek(n,2)+1:nchoosek(n,1)+1+nchoosek(n,2)+nchoosek(n,3))= nchoose3CPM;

CPMA=zeros(n+1,m*sn);
rCPMA=zeros(1,m*sn);
for ai=1:sn
    CPMA(:,m*(ai-1)+1:m*ai)=CPM(:,:,ai);
    rCPMA(1,m*(ai-1)+1:m*ai)=diag(r'*CPMA(2:n+1,m*(ai-1)+1:m*ai))';
end

%target coefficient matrix of requirements in Fluid Model
ITarget = zeros(n*m,m);
for j = 1:m
    ITarget((j-1)*n+1:j*n,j)=ones(n,1);
end
CPMATarget = repmat(CPMA(2:n+1,:),m,1).*repmat(ITarget,1,sn);

%algorithm performance setup
tub = zeros(length(CV),length(LF));%theoritical upperbound of the total FV

%the ratio between the expected FV and theoritical upperbound
tud = zeros(length(CV),length(LF));%DWO Policy
tul = zeros(length(CV),length(LF));%Fluid Policy
tulr = zeros(length(CV),length(LF));%Fluid-R Policy
tulir = zeros(length(CV),length(LF));%Fluid-I-R Policy
tuler = zeros(length(CV),length(LF));%Fluid-E-R Policy

%the ratio between the standard deviation of FV and theoritical upperbound
tsud = zeros(length(CV),length(LF));%DWO Policy
tsul = zeros(length(CV),length(LF));%Fluid Policy
tsulr = zeros(length(CV),length(LF));%Fluid-R Policy
tsulir = zeros(length(CV),length(LF));%Fluid-I-R Policy
tsuler = zeros(length(CV),length(LF));%Fluid-E-R Policy

numAlg = 5;%number of policies

%record changing number of clicks by periods
npcdChange = zeros(n,m,T,S,length(LF),length(CV));%DWO Policy
npclChange = zeros(n,m,T,S,length(LF),length(CV));%Fluid Policy
npclrChange = zeros(n,m,T,S,length(LF),length(CV));%Fluid-R Policy
npclirChange = zeros(n,m,T,S,length(LF),length(CV));%Fluid-I-R Policy
npclerChange = zeros(n,m,T,S,length(LF),length(CV));%Fluid-E-R Policy

alphSummary = zeros(n,m,length(CV),length(LF));%summary of optimal targets

npcSummary = zeros(n,m,S,length(CV),length(LF),numAlg);%summary of total number of clicks

%times # of resolving/infrequently resolving
numResolving = ceil(log10(log10(T))/log10(6/5))+1;
lirInterval = zeros(numResolving,1);
for i=1:numResolving
    lirInterval(i) = T-T^((5/6)^(i-1));
end
lirInterval = ceil(lirInterval);

%computing time record
computingTime = zeros(numAlg,1);

for cvi = 1:length(CV)
    %customer arrival distribution
    p = pSummary(:,cvi);
    cp = p;%cumulative proportion of arrival
    for pi = 2:m
        cp(pi) = cp(pi)+cp(pi-1);
    end

    %rhs of target optimization
    rhsTarget = zeros(n*m+n+m,1);
    for j = 1:m
        rhsTarget((j-1)*n+1:(j-1)*n+n) = p(j);
        rhsTarget(n*m+j) = p(j);
    end

    %fairness coefficient matrix in constraints
    fairnessMatrix = zeros(n*m*(m-1)/2,n*m);
    for i=1:n
        for j=1:m
            for k=j+1:m
                fairnessMatrix((i-1)*m*(m-1)/2+(2*m-j)*(j-1)/2+k-j,(j-1)*n+i)=p(k);%p(j)*p(k);
                fairnessMatrix((i-1)*m*(m-1)/2+(2*m-j)*(j-1)/2+k-j,(k-1)*n+i)=-p(j);%-p(j)*p(k);
            end
        end
    end

    for lfi = 1:length(LF)
        fprintf('Case %d is running with CP=%g and LF=%g. The number of left cases is %d!\n',(cvi-1)*length(LF)+lfi, CV(cvi), LF(lfi), length(CV)*length(LF)-(cvi-1)*length(LF)-lfi);%show the case and number of remaining cases

        mu=round(tratio/LF(lfi));%generate budget of ads
        bdg=mu*ones(n,1);%budget of ads
        rhsTarget(n*m+m+1:n*m+m+n) = bdg/T;%average budget over time

        %Comment the below codes if load ``AdSetting.mat''
        %Construct the requirements eta
        clear model;
        model.A = sparse([lhsTarget zeros(n*m+n+m,n*m*(m-1)/2);  fairnessMatrix -eye(n*m*(m-1)/2);-fairnessMatrix -eye(n*m*(m-1)/2)]);%;CPMAT
        model.rhs=[rhsTarget' zeros(1,n*m*(m-1))];%T0*cr(cri)*alphcL'
        model.sense=[repmat('<',1,n*m+n+m+n*m*(m-1)) ];%repmat('>',1,n*m)
        model.obj=[reshape(rTarget,1,n*m) -lambda*ones(1,n*m*(m-1)/2)];%[r(:,1)'*CPMA(2:n+1,:) -lambda*ones(1,n*m*(m-1)/2)];
        model.modelsense='max';
        model.lb = [zeros(1,n*m) repmat(-inf,1,n*m*(m-1)/2)];
        params.outputflag = 0;
        result = gurobi(model,params);
        eta = reshape(result.x(1:n*m),n,m);
        eta = eta.*unifrnd(0,1,n,m);
        etaTarget(:,:,cvi,lfi) = round(eta*T);%make requirements to integer
        %Comment the above codes if load ``AdSetting.mat''

        eta = etaTarget(:,:,cvi,lfi)/T;%Do not comment this line

        %Find the targets
        tic;
        clear model;
        model.A = sparse([lhsTarget zeros(n*m+n+m,n*m*(m-1)/2);  fairnessMatrix -eye(n*m*(m-1)/2);-fairnessMatrix -eye(n*m*(m-1)/2)]);%;CPMAT
        model.rhs=[rhsTarget' zeros(1,n*m*(m-1))];%T0*cr(cri)*alphcL'
        model.sense=[repmat('<',1,n*m+n+m+n*m*(m-1)) ];%repmat('>',1,n*m)
        model.obj=[reshape(r,1,n*m) -lambda*ones(1,n*m*(m-1)/2)];%[r(:,1)'*CPMA(2:n+1,:) -lambda*ones(1,n*m*(m-1)/2)];
        model.modelsense='max';
        model.lb = [reshape(eta,1,n*m) repmat(-inf,1,n*m*(m-1)/2)];
        params.outputflag = 0;
        result = gurobi(model,params);
        alph = reshape(result.x(1:n*m),n,m);
        alphABS = reshape(result.x(n*m+1:n*m+n*m*(m-1)/2),n,m*(m-1)/2);
        tub(cvi,lfi) = (sum(sum(r.*alph))-lambda*sum(sum(alphABS)))*T;
        alphSummary(:,:,cvi,lfi) = alph;
        computingTime(5)=computingTime(5)+toc;

        %Find the Fluid randomized policy
        tic;
        clear model;
        model.A=sparse([CPMA(2:n+1,:) zeros(n,n*m*(m-1)/2);repmat(eye(m),1,sn) zeros(m,n*m*(m-1)/2);CPMATarget zeros(n*m,n*m*(m-1)/2);  fairnessMatrix*CPMATarget eye(n*m*(m-1)/2);-fairnessMatrix*CPMATarget eye(n*m*(m-1)/2)]);%;CPMAT
        model.rhs=[bdg' T*p' reshape(eta*T,n*m,1)' zeros(1,n*m*(m-1))];
        model.sense=[repmat('<',1,n) repmat('<',1,m) repmat('>',1,n*m) repmat('>',1,n*m*(m-1))];
        model.obj=[rCPMA -lambda*ones(1,n*m*(m-1)/2)];
        model.modelsense='max';
        model.lb = [zeros(1,sn*m+n*m*(m-1)/2)];
        params.outputflag = 0;
        result = gurobi(model,params);
        ps=result.x(1:sn*m);
        pspo=[CPMA(1,ps>0)', mod(find(ps>0),m),ps(ps>0)];
        for i=1:length(pspo(:,1))
            if pspo(i,2)==0
                pspo(i,2)=m;
            else
            end
        end
        pspo(:,3)=pspo(:,3)./(p(pspo(:,2))*T);
        psty=zeros(length(pspo(:,1)),2,m);
        for i=1:m
            a=find(pspo(:,2)==i);
            psty(1:length(a),:,i)=pspo(a,[1,3]);
            for j=2:length(pspo(:,1))
                psty(j,2,i)=psty(j,2,i)+psty(j-1,2,i);
            end
        end
        computingTime(1)=computingTime(1)+toc;

        %set cumulative revenue
        rd=zeros(T+1,S);
        rl=zeros(T+1,S);
        rlr=zeros(T+1,S);
        rlir = zeros(T+1,S);
        rler = zeros(T+1,S);

        %set total revenue
        rsd = zeros(1,S);
        rsl = zeros(1,S);
        rslr = zeros(1,S);
        rslir = zeros(1,S);
        rsler = zeros(1,S);

        %set fairness metric
        frsd = zeros(S,1);
        frsl = zeros(S,1);
        frslr = zeros(S,1);
        frslir = zeros(S,1);
        frsler = zeros(S,1);

        for s=1:S
            %Choice Process
            debt=ones(n,m);%set debt of DWO

            npcd=zeros(n,m);%set # of ads by customer in OCO
            npcl=zeros(n,m);%set # of ads by customer in LP
            npclr=zeros(n,m);%set # of ads by customer in LP Resolving
            npclrL=zeros(n*m,1);
            npclir=zeros(n,m);%set # of ads by customer in LP Infreqently Resolving
            npclirL=zeros(n*m,1);
            npcler=zeros(n,m);%set # of ads by customer in LP Everty-period Resolving
            npclerL=zeros(n*m,1);

            epsilon=-evrnd(0.5772,1,n,T);%Gumble Distribution of each ads
            epsilon0=-evrnd(0.5772,1,1,T);%Gumble Distribution of Outside Option

            for t=1:T
                %generate customer
                gty=unifrnd(0,1);%generate type randomly
                ty=find(cp-gty*ones(m,1)>=0, 1);%choose type
                utility=[epsilon0(t);V(:,ty)+epsilon(:,t)];%utility of customer in t

                %DWO policy
                tic;
                clear model;
                if bdg-sum(npcd,2)<=0
                    pcd=0;
                else
                    zeroI=ones(n,1);
                    zeroI(bdg-sum(npcd,2)<=0)=0;
                    model.A=sparse([eV(:,ty)' 1;ones(1,n) -K;eye(n) -ones(n,1);eye(n) zeros(n,1);eye(n) -zeroI]);
                    model.obj=[debt(:,ty)'.*eV(:,ty)' 0];
                    model.modelsense='max';
                    model.rhs=[1 0 zeros(1,n) zeros(1,n) zeros(1,n)];
                    model.sense=['=' '<' repmat('<',1,n) repmat('>',1,n) repmat('<',1,n)];
                    params.outputflag = 0;
                    result = gurobi(model,params);
                    y=result.x(1:n);
                    z=result.x(n+1);
                    xd=y/z;%assortment
                    eud=exp(utility).*[1;xd];%utility of assortment
                    pcd=find(eud==max(eud))-1;%pcd ad is purchased
                end
                if pcd==0%outside option is choosed
                    rd(t+1,s)=rd(t,s);%revenue is zero
                else
                    npcd(pcd,ty)=npcd(pcd,ty)+1;%one ad pcd is purchased by ty
                    rd(t+1,s)=r(pcd,ty)+rd(t,s);%add the revenue of pcd
                end
                debt=alph*t-npcd;%update the debt
                computingTime(5)=computingTime(5)+toc;

                %Fluid Policy
                tic;
                if bdg-sum(npcl,2)<=0
                    pcl=0;
                else
                    rp=unifrnd(0,1);
                    rs=find(psty(:,2,ty)-rp*ones(length(pspo(:,1)),1)>=0, 1);
                    if isempty(rs)
                        xl=zeros(1,n);
                    else
                        xl=bitget(psty(rs,1,ty),1:n);
                    end
                    xl=xl'.*(bdg-sum(npcl,2)>0);
                    eul=exp(utility).*[1;xl];%utility of assortment
                    pcl=find(eul==max(eul))-1;%which ad is purchased
                end
                if pcl==0%outside option is choosed
                    rl(t+1,s)=rl(t,s);%revenue is zero
                else
                    npcl(pcl,ty)=npcl(pcl,ty)+1;%one ad pce is purchased
                    rl(t+1,s)=r(pcl,ty)+rl(t,s);%add the revenue of pce
                end
                computingTime(1)=computingTime(1)+toc;

                %Fluid Resolving
                %Fluid resolving prepare
                tic;
                if mod(t-1, floor(T/numResolving))==0
                    clear model;
                    model.A=sparse([CPMA(2:n+1,:) zeros(n,n*m*(m-1)/2);repmat(eye(m),1,sn) zeros(m,n*m*(m-1)/2);CPMATarget zeros(n*m,n*m*(m-1)/2);  fairnessMatrix*CPMATarget eye(n*m*(m-1)/2);-fairnessMatrix*CPMATarget eye(n*m*(m-1)/2)]);%;CPMAT
                    model.rhs=[(bdg-sum(npclr,2))' (T-t+1)*p' reshape(eta*T-npclr,n*m,1)' -(fairnessMatrix*npclrL)' (fairnessMatrix*npclrL)'];%T*cr(cri)*alphcL'-npclrL'
                    model.sense=[repmat('<',1,n) repmat('<',1,m) repmat('>',1,n*m) repmat('>',1,n*m*(m-1))];%repmat('>',1,n*m)
                    model.obj=[rCPMA -lambda*ones(1,n*m*(m-1)/2)];%[r(:,1)'*CPMA(2:n+1,:) -lambda*ones(1,n*m*(m-1)/2)];
                    model.modelsense='max';
                    model.lb = [zeros(1,sn*m+n*m*(m-1)/2)];
                    params.outputflag = 0;
                    result = gurobi(model,params);
                    if strcmp(result.status, 'OPTIMAL')
                        psr=result.x(1:sn*m);
                        pspor=[CPMA(1,psr>0)', mod(find(psr>0),m),psr(psr>0)];
                        for i=1:length(pspor(:,1))
                            if pspor(i,2)==0
                                pspor(i,2)=m;
                            else
                            end
                        end
                        pspor(:,3)=pspor(:,3)./(p(pspor(:,2))*(T-t+1));
                        pstyr=zeros(length(pspor(:,1)),2,m);
                        for i=1:m
                            a=find(pspor(:,2)==i);
                            pstyr(1:length(a),:,i)=pspor(a,[1,3]);
                            for j=2:length(pspor(:,1))
                                pstyr(j,2,i)=pstyr(j,2,i)+pstyr(j-1,2,i);
                            end
                        end
                    else
                    end
                else
                end
                %generate click
                if bdg-sum(npclr,2)<=0
                    pclr=0;
                else
                    rpr=unifrnd(0,1);
                    rsr=find(pstyr(:,2,ty)-rpr*ones(length(pspor(:,1)),1)>=0, 1);
                    if isempty(rsr)
                        xlr=zeros(1,n);
                    else
                        xlr=bitget(pstyr(rsr,1,ty),1:n);
                    end
                    xlr=xlr'.*(bdg-sum(npclr,2)>0);
                    eulr=exp(utility).*[1;xlr];%utility of assortment
                    pclr=find(eulr==max(eulr))-1;%which ad is purchased
                end
                if pclr==0%outside option is choosed
                    rlr(t+1,s)=rlr(t,s);%revenue is zero
                else
                    npclr(pclr,ty)=npclr(pclr,ty)+1;%one ad pce is purchased
                    rlr(t+1,s)=r(pclr,ty)+rlr(t,s);%add the revenue of pce
                    npclrL((ty-1)*n+pclr)=npclrL((ty-1)*n+pclr)+1;
                end
                computingTime(2)=computingTime(2)+toc;

                %Fluid Infrequently Resolving
                %Fluid infrequently resolving prepare
                tic;
                if ismember(t-1,lirInterval)==1
                    clear model;
                    model.A=sparse([CPMA(2:n+1,:) zeros(n,n*m*(m-1)/2);repmat(eye(m),1,sn) zeros(m,n*m*(m-1)/2); CPMATarget zeros(n*m,n*m*(m-1)/2); fairnessMatrix*CPMATarget eye(n*m*(m-1)/2);-fairnessMatrix*CPMATarget eye(n*m*(m-1)/2)]);%;CPMAT
                    %model.obj=[r(:,ty)'.*sum(adebt,2)'.*eV(:,ty)' 0];
                    model.obj=[rCPMA -lambda*ones(1,n*m*(m-1)/2)];%[r(:,1)'*CPMA(2:n+1,:) -lambda*ones(1,n*m*(m-1)/2)];
                    model.modelsense='max';
                    model.rhs=[(bdg-sum(npclir,2))' (T-t+1)*p' reshape(eta*T-npclir,n*m,1)' (fairnessMatrix*npclirL)' -(fairnessMatrix*npclirL)'];%T*cr(cri)*alphcL'-npclrL'
                    model.sense=[repmat('<',1,n) repmat('<',1,m) repmat('>',1,n*m) repmat('>',1,n*m*(m-1))];%repmat('>',1,n*m)
                    model.lb = [zeros(1,sn*m+n*m*(m-1)/2)];
                    params.outputflag = 0;
                    result = gurobi(model,params);
                    if strcmp(result.status, 'OPTIMAL')
                        psir=result.x(1:sn*m);
                        pspoir=[CPMA(1,psir>0)', mod(find(psir>0),m),psir(psir>0)];
                        for i=1:length(pspoir(:,1))
                            if pspoir(i,2)==0
                                pspoir(i,2)=m;
                            else
                            end
                        end
                        pspoir(:,3)=pspoir(:,3)./(p(pspoir(:,2))*(T-t+1));
                        pstyir=zeros(length(pspoir(:,1)),2,m);
                        for i=1:m
                            a=find(pspoir(:,2)==i);
                            pstyir(1:length(a),:,i)=pspoir(a,[1,3]);
                            for j=2:length(pspoir(:,1))
                                pstyir(j,2,i)=pstyir(j,2,i)+pstyir(j-1,2,i);
                            end
                        end
                    else
                    end
                else
                end
                %generate click
                if bdg-sum(npclir,2)<=0
                    pclir=0;
                else
                    rpir=unifrnd(0,1);
                    rsir=find(pstyir(:,2,ty)-rpir*ones(length(pspoir(:,1)),1)>=0, 1);
                    if isempty(rsir)
                        xlir=zeros(1,n);
                    else
                        xlir=bitget(pstyir(rsir,1,ty),1:n);
                    end
                    xlir=xlir'.*(bdg-sum(npclir,2)>0);
                    eulir=exp(utility).*[1;xlir];%utility of assortment
                    pclir=find(eulir==max(eulir))-1;%which ad is purchased
                end
                if pclir==0%outside option is choosed
                    rlir(t+1,s)=rlir(t,s);%revenue is zero
                else
                    npclir(pclir,ty)=npclir(pclir,ty)+1;%one ad pce is purchased
                    rlir(t+1,s)=r(pclir,ty)+rlir(t,s);%add the revenue of pce
                    npclirL((ty-1)*n+pclir)=npclirL((ty-1)*n+pclir)+1;
                end
                computingTime(3)=computingTime(3)+toc;


                %Fluid Every-period Resolving
                %Fluid every-period resolving prepare
                tic;
                clear model;
                model.A=sparse([CPMA(2:n+1,:) zeros(n,n*m*(m-1)/2);repmat(eye(m),1,sn) zeros(m,n*m*(m-1)/2); CPMATarget zeros(n*m,n*m*(m-1)/2); fairnessMatrix*CPMATarget eye(n*m*(m-1)/2);-fairnessMatrix*CPMATarget eye(n*m*(m-1)/2)]);
                model.obj=[rCPMA -lambda*ones(1,n*m*(m-1)/2)];
                model.modelsense='max';
                model.rhs=[(bdg-sum(npcler,2))' (T-t+1)*p' reshape(eta*T-npcler,n*m,1)' (fairnessMatrix*npclerL)' -(fairnessMatrix*npclerL)'];%T*cr(cri)*alphcL'-npclrL'
                model.sense=[repmat('<',1,n) repmat('<',1,m) repmat('>',1,n*m) repmat('>',1,n*m*(m-1))];
                model.lb = [zeros(1,sn*m+n*m*(m-1)/2)];
                params.outputflag = 0;
                result = gurobi(model,params);
                if strcmp(result.status, 'OPTIMAL')
                    pser=result.x(1:sn*m);
                    pspoer=[CPMA(1,pser>0)', mod(find(pser>0),m),pser(pser>0)];
                    for i=1:length(pspoer(:,1))
                        if pspoer(i,2)==0
                            pspoer(i,2)=m;
                        else
                        end
                    end
                    pspoer(:,3)=pspoer(:,3)./(p(pspoer(:,2))*(T-t+1));
                    pstyer=zeros(length(pspoer(:,1)),2,m);
                    for i=1:m
                        a=find(pspoer(:,2)==i);
                        pstyer(1:length(a),:,i)=pspoer(a,[1,3]);
                        for j=2:length(pspoer(:,1))
                            pstyer(j,2,i)=pstyer(j,2,i)+pstyer(j-1,2,i);
                        end
                    end
                else
                end
                %generate click and revenue
                if bdg-sum(npcler,2)<=0
                    pcler=0;
                else
                    rper=unifrnd(0,1);
                    rser=find(pstyer(:,2,ty)-rper*ones(length(pspoer(:,1)),1)>=0, 1);
                    if isempty(rser)
                        xler=zeros(1,n);
                    else
                        xler=bitget(pstyer(rser,1,ty),1:n);
                    end
                    xler=xler'.*(bdg-sum(npcler,2)>0);
                    euler=exp(utility).*[1;xler];%utility of assortment
                    pcler=find(euler==max(euler))-1;%which ad is purchased
                end
                if pcler==0%outside option is choosed
                    rler(t+1,s)=rler(t,s);%revenue is zero
                else
                    npcler(pcler,ty)=npcler(pcler,ty)+1;%one ad pce is purchased
                    rler(t+1,s)=r(pcler,ty)+rler(t,s);%add the revenue of pce
                    npclerL((ty-1)*n+pcler)=npclerL((ty-1)*n+pcler)+1;
                end
                computingTime(4)=computingTime(4)+toc;

                npcdChange(:,:,t,s,lfi,cvi) = npcd;
                npclChange(:,:,t,s,lfi,cvi) = npcl;
                npclrChange(:,:,t,s,lfi,cvi) = npclr;
                npclirChange(:,:,t,s,lfi,cvi) = npclir;
                npclerChange(:,:,t,s,lfi,cvi) = npcler;
            end

            rsd(s) = rd(T+1,s);
            rsl(s) = rl(T+1,s);
            rslr(s) = rlr(T+1,s);
            rslir(s) = rlir(T+1,s);
            rsler(s) = rler(T+1,s);

            frsd(s) = sum(abs(fairnessMatrix*reshape(npcd,n*m,1)));
            frsl(s) = sum(abs(fairnessMatrix*reshape(npcl,n*m,1)));
            frslr(s) = sum(abs(fairnessMatrix*reshape(npclr,n*m,1)));
            frslir(s) = sum(abs(fairnessMatrix*reshape(npclir,n*m,1)));
            frsler(s) = sum(abs(fairnessMatrix*reshape(npcler,n*m,1)));

            npcSummary(:,:,s,cvi,lfi,1) = npcl;
            npcSummary(:,:,s,cvi,lfi,2) = npclr;
            npcSummary(:,:,s,cvi,lfi,3) = npclir;
            npcSummary(:,:,s,cvi,lfi,4) = npcler;
            npcSummary(:,:,s,cvi,lfi,5) = npcd;
        end

        tud(cvi,lfi) = mean(rsd-lambda*frsd')/tub(cvi,lfi)*100;
        tul(cvi,lfi) = mean(rsl-lambda*frsl')/tub(cvi,lfi)*100;
        tulr(cvi,lfi) = mean(rslr-lambda*frslr')/tub(cvi,lfi)*100;
        tulir(cvi,lfi) = mean(rslir-lambda*frslir')/tub(cvi,lfi)*100;
        tuler(cvi,lfi) = mean(rsler-lambda*frsler')/tub(cvi,lfi)*100;

        tsud(cvi,lfi) = std(rsd-lambda*frsd')/tub(cvi,lfi)*100;
        tsul(cvi,lfi) = std(rsl-lambda*frsl')/tub(cvi,lfi)*100;
        tsulr(cvi,lfi) = std(rslr-lambda*frslr')/tub(cvi,lfi)*100;
        tsulir(cvi,lfi) = std(rslir-lambda*frslir')/tub(cvi,lfi)*100;
        tsuler(cvi,lfi) = std(rsler-lambda*frsler')/tub(cvi,lfi)*100;
    end
end

%generate the average proportion of unfilled click-through requirements
targetLossAvg = zeros(length(LF),numAlg,length(CV));
for cvi = 1: length(CV)
    for lfi = 1: length(LF)
        for numAlgi = 1:numAlg
            targetLossTotal = zeros(n,m);
            for s = 1:S
                targetLossTmp = min(max(1-npcSummary(:,:,s,cvi,lfi,numAlgi)./etaTarget(:,:,cvi,lfi),0),1);
                targetLossTotal = targetLossTotal+targetLossTmp;
            end
            targetLossAvg(lfi,numAlgi,cvi)=sum(sum(targetLossTotal/S/n/m));
        end
    end
end

%generate quantile of the click-through sample-paths of one specific ad i = 1, lfi = 3, cvi = 4
npcsum1=zeros(S,T,numAlg);
for t=1:T
    for s=1:S
        npcsum1(s,t,1) = sum(npclChange(1,:,t,s,3,4));
        npcsum1(s,t,2) = sum(npclrChange(1,:,t,s,3,4));
        npcsum1(s,t,3) = sum(npclirChange(1,:,t,s,3,4));
        npcsum1(s,t,4) = sum(npclerChange(1,:,t,s,3,4));
        npcsum1(s,t,5) = sum(npcdChange(1,:,t,s,3,4));
    end
end
npcsumQuantile = zeros(3,T,numAlg);
for t=1:T
    for j=1:numAlg
        npcsumQuantile(1,t,j)=quantile(npcsum1(:,t,j),0.1);
        npcsumQuantile(2,t,j)=median(npcsum1(:,t,j));
        npcsumQuantile(3,t,j)=quantile(npcsum1(:,t,j),0.9);
    end
end
fprintf('Main results complete!\n');

%% Plot
figi = 1;
%boxplot
figure(figi)%Figure 2 (a)
boxplot([reshape(tul,length(CV)*length(LF),1),reshape(tulr,length(CV)*length(LF),1),reshape(tulir,length(CV)*length(LF),1),reshape(tuler,length(CV)*length(LF),1),reshape(tud,length(CV)*length(LF),1)]/100,'Labels',{'Fluid','Fluid-R','Fluid-I-R','Fluid-E-R','DWO'})
xlabel('Policy','FontWeight','bold')
ylabel('The ratio of average total value to upper bound','FontWeight','bold')
figi = figi+1;


figure(figi)%Figure 2 (b)
boxplot([reshape(tsul,length(CV)*length(LF),1),reshape(tsulr,length(CV)*length(LF),1),reshape(tsulir,length(CV)*length(LF),1),reshape(tsuler,length(CV)*length(LF),1),reshape(tsud,length(CV)*length(LF),1)]/100,'Labels',{'Fluid','Fluid-R','Fluid-I-R','Fluid-E-R','DWO'})
xlabel('Policy','FontWeight','bold')
ylabel('The ratio of average deviation to upper bound','FontWeight','bold')
figi = figi+1;


figure(figi)%Figure 2 (c)
boxplot([reshape(targetLossAvg(:,1,:),length(CV)*length(LF),1),reshape(targetLossAvg(:,2,:),length(CV)*length(LF),1),reshape(targetLossAvg(:,3,:),length(CV)*length(LF),1),reshape(targetLossAvg(:,4,:),length(CV)*length(LF),1),reshape(targetLossAvg(:,5,:),length(CV)*length(LF),1)],'Labels',{'Fluid','Fluid-R','Fluid-I-R','Fluid-E-R','DWO'})
xlabel('Policy','FontWeight','bold')
ylabel('The proportion of unfilled click-throughs in requirements','FontWeight','bold')
figi = figi+1;

%quantile
figure(figi);%Figure 3 (a)
plot(npcsumQuantile(1,:,1),'-.','LineWidth',2);
hold on;
plot(npcsumQuantile(2,:,1),'-','LineWidth',2);
hold on;
plot(npcsumQuantile(3,:,1),'--','LineWidth',2 );
xlabel('Time Period','FontWeight','bold');
ylabel('Number of Clicks','FontWeight','bold');
legend({'0.1 Quantile','Median','0.9 Quantile'},'Location','northwest','FontWeight','bold');
figi = figi +1;

figure(figi);%Figure 3 (b)
plot(npcsumQuantile(1,:,2),'-.','LineWidth',2);
hold on;
plot(npcsumQuantile(2,:,2),'-','LineWidth',2);
hold on;
plot(npcsumQuantile(3,:,2),'--','LineWidth',2);
xlabel('Time Period','FontWeight','bold');
ylabel('Number of Clicks','FontWeight','bold');
legend({'0.1 Quantile','Median','0.9 Quantile'},'Location','northwest','FontWeight','bold');
figi = figi +1;

figure(figi);%Figure 3 (c)
plot(npcsumQuantile(1,:,3),'-.','LineWidth',2);
hold on;
plot(npcsumQuantile(2,:,3),'-','LineWidth',2);
hold on;
plot(npcsumQuantile(3,:,3),'--','LineWidth',2);
xlabel('Time Period','FontWeight','bold');
ylabel('Number of Clicks','FontWeight','bold');
legend({'0.1 Quantile','Median','0.9 Quantile'},'Location','northwest','FontWeight','bold');
figi = figi +1;

figure(figi);%Figure 3 (d)
plot(npcsumQuantile(1,:,4),'-.','LineWidth',2);
hold on;
plot(npcsumQuantile(2,:,4),'-','LineWidth',2);
hold on;
plot(npcsumQuantile(3,:,4),'--','LineWidth',2);
xlabel('Time Period','FontWeight','bold');
ylabel('Number of Clicks','FontWeight','bold');
legend({'0.1 Quantile','Median','0.9 Quantile'},'Location','northwest','FontWeight','bold');
figi = figi +1;

figure(figi);%Figure 3 (e)
plot(npcsumQuantile(1,:,5),'-.','LineWidth',2);
hold on;
plot(npcsumQuantile(2,:,5),'-','LineWidth',2);
hold on;
plot(npcsumQuantile(3,:,5),'--','LineWidth',2);
xlabel('Time Period','FontWeight','bold');
ylabel('Number of Clicks','FontWeight','bold');
legend({'0.1 Quantile','Median','0.9 Quantile'},'Location','northwest','FontWeight','bold');
figi = figi +1;

%% debt regression to generate Table 3
sampleSize = 30000000;
debtnpcChangeVecd = zeros(sampleSize,2);
debtnpcChangeVecl = zeros(sampleSize,2);
debtnpcChangeVeclr = zeros(sampleSize,2);
debtnpcChangeVeclir = zeros(sampleSize,2);
debtnpcChangeVecler = zeros(sampleSize,2);

sampleIndex = randsample(n*m*(T/2-1)*S*length(LF)*length(CV),sampleSize);
indexSample = zeros(sampleSize,6);
for i = 1:sampleSize
    indexSample(i,6) = mod(sampleIndex(i),length(CV));
    if indexSample(i,6) == 0
        indexSample(i,6) = length(CV);
    end
    indexSample(i,5) = mod(floor(sampleIndex(i)/length(CV)),length(LF));
    if indexSample(i,5) == 0
        indexSample(i,5) = length(LF);
    end
    temp = floor(sampleIndex(i)/length(CV));
    indexSample(i,4) = mod(floor(temp/length(LF)),T/2-1);
    if indexSample(i,4) == 0
        indexSample(i,4) = T/2-1;
    end
    temp = floor(temp/length(LF));
    indexSample(i,3) = mod(floor(temp/T/2-1),S);
    if indexSample(i,3) == 0
        indexSample(i,3) = S;
    end
    temp = floor(temp/T/2-1);
    indexSample(i,2) = mod(floor(temp/S),m);
    if indexSample(i,2) == 0
        indexSample(i,2) = m;
    end
    temp = floor(temp/S);
    indexSample(i,1) = mod(floor(temp/m),n);
    if indexSample(i,1) == 0
        indexSample(i,1) = n;
    end
    debtnpcChangeVecd(i,:)=[alphSummary(indexSample(i,1),indexSample(i,2),indexSample(i,6),indexSample(i,5))-npcdChange(indexSample(i,1),indexSample(i,2),indexSample(i,4),indexSample(i,3),indexSample(i,5),indexSample(i,6))/indexSample(i,4),npcdChange(indexSample(i,1),indexSample(i,2),indexSample(i,4)+1,indexSample(i,3),indexSample(i,5),indexSample(i,6))-npcdChange(indexSample(i,1),indexSample(i,2),indexSample(i,4),indexSample(i,3),indexSample(i,5),indexSample(i,6))];
    debtnpcChangeVecl(i,:)=[alphSummary(indexSample(i,1),indexSample(i,2),indexSample(i,6),indexSample(i,5))-npclChange(indexSample(i,1),indexSample(i,2),indexSample(i,4),indexSample(i,3),indexSample(i,5),indexSample(i,6))/indexSample(i,4),npclChange(indexSample(i,1),indexSample(i,2),indexSample(i,4)+1,indexSample(i,3),indexSample(i,5),indexSample(i,6))-npclChange(indexSample(i,1),indexSample(i,2),indexSample(i,4),indexSample(i,3),indexSample(i,5),indexSample(i,6))];
    debtnpcChangeVeclr(i,:)=[alphSummary(indexSample(i,1),indexSample(i,2),indexSample(i,6),indexSample(i,5))-npclrChange(indexSample(i,1),indexSample(i,2),indexSample(i,4),indexSample(i,3),indexSample(i,5),indexSample(i,6))/indexSample(i,4),npclrChange(indexSample(i,1),indexSample(i,2),indexSample(i,4)+1,indexSample(i,3),indexSample(i,5),indexSample(i,6))-npclrChange(indexSample(i,1),indexSample(i,2),indexSample(i,4),indexSample(i,3),indexSample(i,5),indexSample(i,6))];
    debtnpcChangeVeclir(i,:)=[alphSummary(indexSample(i,1),indexSample(i,2),indexSample(i,6),indexSample(i,5))-npclirChange(indexSample(i,1),indexSample(i,2),indexSample(i,4),indexSample(i,3),indexSample(i,5),indexSample(i,6))/indexSample(i,4),npclirChange(indexSample(i,1),indexSample(i,2),indexSample(i,4)+1,indexSample(i,3),indexSample(i,5),indexSample(i,6))-npclirChange(indexSample(i,1),indexSample(i,2),indexSample(i,4),indexSample(i,3),indexSample(i,5),indexSample(i,6))];
    debtnpcChangeVecler(i,:)=[alphSummary(indexSample(i,1),indexSample(i,2),indexSample(i,6),indexSample(i,5))-npclerChange(indexSample(i,1),indexSample(i,2),indexSample(i,4),indexSample(i,3),indexSample(i,5),indexSample(i,6))/indexSample(i,4),npclerChange(indexSample(i,1),indexSample(i,2),indexSample(i,4)+1,indexSample(i,3),indexSample(i,5),indexSample(i,6))-npclerChange(indexSample(i,1),indexSample(i,2),indexSample(i,4),indexSample(i,3),indexSample(i,5),indexSample(i,6))];
end

ttestdyd = fitlm(debtnpcChangeVecd(:,1),debtnpcChangeVecd(:,2));
ttestdyl = fitlm(debtnpcChangeVecl(:,1),debtnpcChangeVecl(:,2));
ttestdylr = fitlm(debtnpcChangeVeclr(:,1),debtnpcChangeVeclr(:,2));
ttestdylir = fitlm(debtnpcChangeVeclir(:,1),debtnpcChangeVeclir(:,2));
ttestdyler = fitlm(debtnpcChangeVecler(:,1),debtnpcChangeVecler(:,2));

%save('Ad20230220main','-v7.3')


