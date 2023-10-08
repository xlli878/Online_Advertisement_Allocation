clear all;
n=50;%number of ads 50
m=5;%number of customer types 5
S=30;%sample path size 30
KN=5;%cadinality constriant from 1 to 5
timeGurobiDWO = zeros(KN,S);%time record for DWO
timeGurobiFluid = zeros(KN,S);%time record for Fluid
lambda=10;%10;%regularization parameter
uV=5;%upper of distribution of determined utility
tratio=20;%ratio of T to n
T=n*tratio;%total periods

for K = 1:KN
    for s = 1:S
        r=unifrnd(10,50,n,m);%revenue of ads
        rTarget=unifrnd(10,50,n,m);%alternative revenue to generate requirements
        V=unifrnd(0,uV,n,m);%utility of customers
        eV=exp(V);%choice model term

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

        deFromBi=zeros(n,1);
        for i=1:n
            deFromBi(i)=2^(i-1);
        end

        CV=[0.1,1,10,100];
        Table4=zeros(4,9,4);
        cvi=4;
        %Customer Setting
        a1=repmat(CV(cvi),1,m);
        p1 = gamrnd(repmat(a1,1,1),1,1,m);
        p1 = p1 ./ repmat(sum(p1,2),1,m);
        p=p1';
        cp=p;%cumulative proportion
        for pi=2:m
            cp(pi)=cp(pi)+cp(pi-1);
        end

        fairnessMatrix = zeros(n*m*(m-1)/2,n*m);
        for i=1:n
            for j=1:m
                for k=j+1:m
                    fairnessMatrix((i-1)*m*(m-1)/2+(2*m-j)*(j-1)/2+k-j,(j-1)*n+i)=p(k);%p(j)*p(k);
                    fairnessMatrix((i-1)*m*(m-1)/2+(2*m-j)*(j-1)/2+k-j,(k-1)*n+i)=-p(j);%-p(j)*p(k);
                end
            end
        end

        LF=[1.8, 1.5, 1.2, 1];
        lfi=4;

        rhsTarget = zeros(n*m+n+m,1);

        for j = 1:m
            rhsTarget((j-1)*n+1:(j-1)*n+n) = p(j);
            rhsTarget(n*m+j) = p(j);
        end

        %fairness
        fairnessMatrix = zeros(n*m*(m-1)/2,n*m);
        for i=1:n
            for j=1:m
                for k=j+1:m
                    fairnessMatrix((i-1)*m*(m-1)/2+(2*m-j)*(j-1)/2+k-j,(j-1)*n+i)=p(k);%p(j)*p(k);
                    fairnessMatrix((i-1)*m*(m-1)/2+(2*m-j)*(j-1)/2+k-j,(k-1)*n+i)=-p(j);%-p(j)*p(k);
                end
            end
        end

        mu=round(tratio/LF(lfi));%mean budget of ads

        bdg=mu*ones(n,1);%budget of ads
        rhsTarget(n*m+m+1:n*m+m+n) = bdg/T;

        %Construct the targets eta
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
        eta = round(eta*T)/T;

        %Find the targets
        clear model;
        model.A = sparse([lhsTarget zeros(n*m+n+m,n*m*(m-1)/2);  fairnessMatrix -eye(n*m*(m-1)/2);-fairnessMatrix -eye(n*m*(m-1)/2)]);
        model.rhs=[rhsTarget' zeros(1,n*m*(m-1))];
        model.sense=[repmat('<',1,n*m+n+m+n*m*(m-1)) ];
        model.obj=[reshape(r,1,n*m) -lambda*ones(1,n*m*(m-1)/2)];
        model.modelsense='max';
        model.lb = [reshape(eta,1,n*m) repmat(-inf,1,n*m*(m-1)/2)];
        params.outputflag = 0;
        result = gurobi(model,params);
        timeGurobiDWO(K,s) = result.runtime;

        % Fluid
        %construct Fluid decision set
        sn = 1;
        for numkk = 1:K
            sn = sn +nchoosek(n,numkk);
        end

        CPM=zeros(n+1,m,sn);

        if K>=1
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
            CPM(:,:,2:nchoosek(n,1)+1)= nchoose1CPM;
            clear nchoose1CPM;
        else
        end

        if K>=2
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
            CPM(:,:,nchoosek(n,1)+1+1:nchoosek(n,1)+1+nchoosek(n,2))= nchoose2CPM;
            clear nchoose2CPM;
        else
        end

        if K>=3
            nchoose3CPM = zeros(n+1,m,nchoosek(n,3));
            nchoose3t=1;
            for nchoose3i=1:n-3+1
                for nchoose3j=nchoose3i+1:n-3+2
                    for nchoose3k=nchoose3j+1:n-3+3
                        nchoose3ss=zeros(n,1);
                        nchoose3ss(nchoose3i)=1;
                        nchoose3ss(nchoose3j)=1;
                        nchoose3ss(nchoose3k)=1;
                        for tyi=1:m
                            nchoose3CPM(1,tyi,nchoose3t)=nchoose3ss'*deFromBi;
                            nchoose3CPM(2:n+1,tyi,nchoose3t)=nchoose3ss.*eV(:,tyi)/(1+nchoose3ss'*eV(:,tyi));
                        end
                        nchoose3t=nchoose3t+1;
                    end
                end
            end
            CPM(:,:,nchoosek(n,1)+1+nchoosek(n,2)+1:nchoosek(n,1)+1+nchoosek(n,2)+nchoosek(n,3))= nchoose3CPM;
            clear nchoose3CPM;
        else
        end

        if K>=4
            nchoose4CPM = zeros(n+1,m,nchoosek(n,4));
            nchoose4t=1;
            for nchoose4i=1:n-4+1
                for nchoose4j=nchoose4i+1:n-4+2
                    for nchoose4k=nchoose4j+1:n-4+3
                        for nchoose4l=nchoose4k+1:n-4+4
                            nchoose4ss=zeros(n,1);
                            nchoose4ss(nchoose4i)=1;
                            nchoose4ss(nchoose4j)=1;
                            nchoose4ss(nchoose4k)=1;
                            nchoose4ss(nchoose4l)=1;
                            for tyi=1:m
                                nchoose4CPM(1,tyi,nchoose4t)=nchoose4ss'*deFromBi;
                                nchoose4CPM(2:n+1,tyi,nchoose4t)=nchoose4ss.*eV(:,tyi)/(1+nchoose4ss'*eV(:,tyi));
                            end
                            nchoose4t=nchoose4t+1;
                        end
                    end
                end
            end
            CPM(:,:,nchoosek(n,1)+1+nchoosek(n,2)+nchoosek(n,3)+1:nchoosek(n,1)+1+nchoosek(n,2)+nchoosek(n,3)+nchoosek(n,4))= nchoose4CPM;
            clear nchoose4CPM;
        else
        end

        if K<5%K>=5
            % nchoose5CPM = zeros(n+1,m,nchoosek(n,5));
            % nchoose5t=1;
            % for nchoose5i=1:n-5+1
            %     for nchoose5j=nchoose5i+1:n-5+2
            %         for nchoose5k=nchoose5j+1:n-5+3
            %             for nchoose5l=nchoose5k+1:n-5+4
            %                 for nchoose5m=nchoose5l+1:n-5+5
            %                     nchoose5ss=zeros(n,1);
            %                     nchoose5ss(nchoose5i)=1;
            %                     nchoose5ss(nchoose5j)=1;
            %                     nchoose5ss(nchoose5k)=1;
            %                     nchoose5ss(nchoose5l)=1;
            %                     nchoose5ss(nchoose5m)=1;
            %                     for tyi=1:m
            %                         nchoose5CPM(1,tyi,nchoose5t)=nchoose5ss'*deFromBi;
            %                         nchoose5CPM(2:n+1,tyi,nchoose5t)=nchoose5ss.*eV(:,tyi)/(1+nchoose5ss'*eV(:,tyi));
            %                     end
            %                     nchoose5t=nchoose5t+1;
            %                 end
            %             end
            %         end
            %     end
            % end
            % CPM(:,:,nchoosek(n,1)+1+nchoosek(n,2)+nchoosek(n,3)+nchoosek(n,4)+1:nchoosek(n,1)+1+nchoosek(n,2)+nchoosek(n,3)+nchoosek(n,4)+nchoosek(n,5))= nchoose5CPM;
            % clear nchoose5CPM;
            % else
            % end

            CPMA=zeros(n+1,m*sn);
            rCPMA=zeros(1,m*sn);
            for ai=1:sn
                CPMA(:,m*(ai-1)+1:m*ai)=CPM(:,:,ai);
                rCPMA(1,m*(ai-1)+1:m*ai)=diag(r'*CPMA(2:n+1,m*(ai-1)+1:m*ai))';
            end

            ITarget = zeros(n*m,m);
            for j = 1:m
                ITarget((j-1)*n+1:j*n,j)=ones(n,1);
            end
            CPMATarget = repmat(CPMA(2:n+1,:),m,1).*repmat(ITarget,1,sn);

            %Find the randomized policy of Fluid
            clear model;
            model.A=sparse([CPMA(2:n+1,:) zeros(n,n*m*(m-1)/2);repmat(eye(m),1,sn) zeros(m,n*m*(m-1)/2);CPMATarget zeros(n*m,n*m*(m-1)/2);  fairnessMatrix*CPMATarget eye(n*m*(m-1)/2);-fairnessMatrix*CPMATarget eye(n*m*(m-1)/2)]);%;CPMAT
            model.rhs=[bdg' T*p' reshape(eta*T,n*m,1)' zeros(1,n*m*(m-1))];
            model.sense=[repmat('<',1,n) repmat('<',1,m) repmat('>',1,n*m) repmat('>',1,n*m*(m-1))];
            model.obj=[rCPMA -lambda*ones(1,n*m*(m-1)/2)];
            model.modelsense='max';
            model.lb = [zeros(1,sn*m+n*m*(m-1)/2)];
            params.outputflag = 0;
            result = gurobi(model,params);
            timeGurobiFluid(K,s) = result.runtime;
            clear CPM CPMATarget CPMA rCPMA;%clear space to improve efficiency
        else
        end

    end
end
timeAvgDWO = mean(timeGurobiDWO,2);%average time record for DWO
timeAvgFluid = mean(timeGurobiFluid,2);%average time record for Fluid

%save('AdTime','-v7.3')