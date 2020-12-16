classdef auxi
    methods(Static)
        
        
        %% Creates parameter for network with E and I clusters
        % from ClustersOption='EI'
        % Luca Mazzucato December 2020
        
        function create_params(paramsfile,Opt)
            %----------
            % NETWORKS
            %----------
            % Start and end of trial (in units of seconds)
            Sim.t_Start=-1;
            Sim.t_End=1;
            Sim.dt_step=0.0001; % integration step (s)
            
            %------------------
            % TIME CONSTANTS
            %------------------
            tau_arp = .005;  % refractory period
            tau_i = .02;     % inh membrane time
            tau_e = .02;	 % exc membrane time
            tausyn_e=0.005;  % exc synaptic time
            tausyn_i=0.005;  % inh synaptic time
            if strcmp(Opt,'EI')
                %------------------
                % NETWORK OPTIONS
                %------------------
                Network.clusters={'EE','EI','IE','II'}; % all weights are clustered
                % Network.clust='hom'; % homogeneous EE cluster size
                Network.clust='het'; % heterogeneous EE cluster size
                Network.clust_std=0.2; % het cluster size: cluster size is sampled from a gaussian with SD=Ncluster*Network.std
                Network.clustEI='hom'; % EI clusters: homogeneous='hom', heterogeneous='het'
                Network.clustIE='hom'; % IE clusters: homogeneous='hom', heterogeneous='het'
                Network.clustII='hom'; % II clusters: homogeneous='hom', heterogeneous='het'
                % Network.clust_syn='';
                N=2000; % network size
                N_e = N*4/5; % exc neurons
                N_i = N/5; % inh neurons
                Scale=(1000/N)^(1/2);
                %------------------
                % SYNAPTIC WEIGHTS
                %------------------
                % syn weights are drawn from a gaussian distribution with std delta and
                % mean Jab
                Jee = Scale*0.02;
                Jii = 1.*Scale*0.12; %     Jii = Scale*0.06;
                Jie = 2*Scale*0.010; %     Jei 3 Scale*0.045;
                Jei = 3.*Scale*0.02;
                %------------------
                % CLUSTER PARAMETERS
                %------------------
                delta=0.2;% SD of synaptic weights distribution, eg: ~Jee*(1+delta*randn(N_e))
                Network.delta = delta; % SD of synaptic weights: Jee*(1+delta*randn(N_e)) Larger delta -> clusters look more async in the high state
                Network.deltaEI = delta; % SD of synaptic weights: Jee*(1+delta*randn(N_e)) Larger delta -> clusters look more async in the high state
                Network.deltaIE = delta; % SD of synaptic weights: Jee*(1+delta*randn(N_e)) Larger delta -> clusters look more async in the high state
                Network.deltaII = delta; % SD of synaptic weights: Jee*(1+delta*randn(N_e)) Larger delta -> clusters look more async in the high state
                Jplus = 14; % EE intra-cluster potentiation factor
                Network.factorEI = 10; % EI intra-cluster potentiation factor
                Network.factorIE = 8; % IE intra-cluster potentiation factor
                Network.factorII = 5; % II intra-cluster potentiation factor
                %
                bgr=0.1; % fraction of background excitatory neurons (unclustered)
                Network.bgrI=0.1; % fraction of background neurons
                Ncluster=80; % average # neurons per cluster
                p = round(N_e*(1-bgr)/Ncluster); % # of clusters
                f = (1-bgr)/p; % fraction of E neurons per cluster
                Network.fI = (1-Network.bgrI)/p;       % fraction of I neurons per cluster
                %------------------
                % THRESHOLDS
                %------------------
                theta_e=1.42824;% exc threshold potential
                theta_i=0.74342;% inh threshold potentials
                % reset potentials
                He = 0;%
                Hi = 0;%
                % EXTERNAL CURRENT
                % default external currents
                ni_ext = 5; % 7;
            elseif strcmp(Opt,'E')
                % Creates parameter for network with E but not I clusters
                %------------------
                % NETWORK OPTIONS
                %------------------
                % Network.clust='hom'; % homogeneous cluster size
                Network.clust='hom'; % heterogeneous cluster size
                Network.clust_std=0.01; % het cluster size: cluster size is sampled from a gaussian with SD=Ncluster*Network.std
                N=2000; % network size
                N_e = N*4/5; % E neurons
                N_i = N/5; % I neurons
                Scale=(5000/N)^(1/2);
                %------------------
                % SYNAPTIC WEIGHTS
                %------------------
                % syn weights are drawn from a gaussian distribution with std delta and
                % mean Jab
                Jee = Scale*0.015; % mean Jee weight
                Jii = (5/2)^(1/2)*Scale*0.06;
                Jie = Scale*0.02;
                Jei = (5/2)^(1/2)*Scale*0.045;
                delta = 0.1; % SD of synaptic weights distribution, eg: ~Jee*(1+delta*randn(N_e))
                Network.delta = delta; % SD of synaptic weights: Jee*(1+delta*randn(N_e)) Larger delta -> clusters look more async in the high state
                Network.deltaEI = delta; % SD of synaptic weights: Jee*(1+delta*randn(N_e)) Larger delta -> clusters look more async in the high state
                Network.deltaIE = delta; % SD of synaptic weights: Jee*(1+delta*randn(N_e)) Larger delta -> clusters look more async in the high state
                Network.deltaII = delta; % SD of synaptic weights: Jee*(1+delta*randn(N_e)) Larger delta -> clusters look more async in the high state
                %------------------
                % CLUSTER PARAMETERS
                %------------------
                Jplus = 9; % intra-cluster potentiation factor for E clusters
                bgr=0.1; % fraction of background excitatory neurons (unclustered)
                Ncluster=100; % average # neurons per cluster
                p = round(N_e*(1-bgr)/Ncluster); % # of clusters
                f = (1-bgr)/p;
                %------------------
                % THRESHOLDS
                %------------------
                theta_e=4; % exc threshold potential
                theta_i=4; % inh threshold potentials
                % reset potentials
                He = 0;%
                Hi = 0;%
                % EXTERNAL CURRENT
                % default external currents
                ni_ext = 7; % 7;
            end
            %------------------
            % CONNECTIVITY PARAMETERS
            %------------------
            Cee = N_e*0.2; % # presynaptic neurons
            Cie = N_e*0.5; %
            Cii = N_i*0.5; %
            Cei = N_i*0.5; %
            %------------------
            % EXTERNAL BIAS
            %------------------
            % external input parameters, eg: external current given by mu_e_ext=Cext*Jee_ext*ni_ext
            Cext = (N_e)*0.2; % # presynaptic external neurons
            Jie_ext=0.8*Scale*0.0915;% external input synaptic strengths
            Jee_ext=0.8*Scale*0.1027; %
            mu_E0=Cext*Jee_ext*ni_ext;
            mu_I0=Cext*Jie_ext*ni_ext;
            % random ext current, different in each trial
            Mu=[mu_E0*(ones(N_e,1)+(0.1/2)*(2*rand([N_e,1])-1)); ...
                mu_I0*(ones(N_i,1)+(0.05/2)*(2*rand([N_i,1])-1))];     % bias
            
            
            %----------------
            % DEFAULT STIMULI
            %----------------
            % STIMULUS
            Stimulus.input='Const'; % constant external current
            scnt=0;
            % TASTE (specific stimulus)
            scnt=scnt+1;
            feat(scnt).name='US'; % unconditioned stimulus (taste)
            feat(scnt).interval=[0 Sim.t_End]; % stimulus interval
            gain=0.1; % stimulus value at 1 s
            feat(scnt).gain=gain;
            feat(scnt).profile=@(t)t; % time course of stimulus, eg a linear ramp
            feat(scnt).selectivity='mixed'; % random half of clusters are selective for each stimulus
            feat(scnt).selective=rand(1,p)<0.5; % US selective clusters
            feat(scnt).connectivity=0.5; % fraction of selective neurons within a selective cluster
            % ANTICIPATORY CUE
            scnt=scnt+1;
            feat(scnt).name='CSgauss'; % conditioned stimulus (cue) with "quenched" noise
            feat(scnt).interval=[-0.5 Sim.t_End]; % stimulus interval
            gain=0.1; % SD of quenched noise across neurons
            feat(scnt).gain=gain;
            %             tau_cue=[0.5,1]; % rise and decay time of double exp cue time course
            %             feat(scnt).profile=@(t)(1/(tau_cue(2)-tau_cue(1)))*(exp(-t/tau_cue(2))-exp(-t/tau_cue(1))); % double exp profile time course
            feat(scnt).profile=@(t)1;
            feat(scnt).selectivity='exc'; % targets exc neurons only
            feat(scnt).selective=ones(1,p); % CS targets all clusters
            feat(scnt).connectivity=0.50; % fraction of neurons targeted within each cluster
            %
            Stimulus.feat=feat;
            %------------------------------------------------------------------------
            % PLOT PARAMETERS: ------------------------------------------------
            %------------------------------------------------------------------------/
            % PLOTS
            Sim.Plotf=0;
            Sim.plot_length=Sim.t_End-Sim.t_Start; % length of plot intervals
            % indices of ensemble units to store
            exc=randperm(N_e);
            inh=N_e+randperm(N_i);
            Sim.ind_p=[exc(1) inh(1)]; % choosing neuron index for membrane potential plot (one E and one I)
            Sim.weights_save='off'; % save weight matrix: 'Yes'
            extra='';
            save(paramsfile,'ni_ext','tau_arp','tau_i','tau_e','theta_e',...
                'theta_i','delta','f','Jee','Jii','Jie','Jei','Jee_ext','Jie_ext',...
                'Jplus','He','Hi','N_e',...
                'N_i','Cee','Cie','Cei','Cii','Cext','p','Sim','Network','Stimulus',...
                'tausyn_e','tausyn_i','extra','Mu','paramsfile');
            fprintf('Network parameters saved in %s\n',paramsfile);
        end
        
        %% function [J, params]=SynWeights(params)
        %
        % OUTPUT
        %       J                =synaptic matrix
        %       params.popsize      =array of dim # pops, how many exc neurons in each
        %                         pop
        %       params.clustermatrix=matrix of dimension # pops x # clusters, each row shows
        %                   which clusters belong to that pop (1s and 0s)
        %
        % Luca Mazzucato December 2020
        function [J, params]=fun_SynWeights(paramsfile,Opt)
            
            % LOAD PARAMETERS
            params=load(paramsfile);
            utils.v2struct(params);
            Network=params.Network;
            utils.v2struct(Network);
            Sim=params.Sim;
            %-----------------------
            % PARAMETERS VALUES
            %-----------------------
            numfig=1;
            Next=N_e; % external units
            % CLUSTERS
            Q=p; % number of clusters
            %-----------------------
            % SYNAPTIC WEIGHTS
            %-----------------------
            % WEIGHTS
            % depression
            gam=1/(2-f*(Q+1));%
            Jminus = 1.-gam*f*(Jplus-1.);
            params.Jminus = Jminus;
            %
            jee=Jee;
            jee_out=Jminus*Jee; % intra-cluster potentiation
            jee_in=Jplus*Jee; % inter-cluster depression
            jei=-Jei;
            jie=Jie;
            jii=-Jii;
            % connection probability
            pee=Cee/N_e;
            pie=Cie/N_e;
            pei=Cei/N_i;
            pii=Cii/N_i;
            pext=Cext/Next;
            peeout=pee;
            peein=pee;
            peiout=pei;
            peiin=pei;
            pieout=pie;
            piein=pie;
            piiout=pii;
            piiin=pii;
            %
            fprintf('  --- Jplus=%0.03g, Jminus=%0.03g\n',Jplus,Jminus);
            %----------------------------
            % SYNAPTIC MATRIX
            %----------------------------
            % generate a distribution of synaptic weights with mean J and
            % variance delta^2 J^2
            peeout=pee;
            peein=pee;
            % check #clusters and coding level are consistent
            NcUnits=round(f*N_e);    %  number of Exc units per cluster
            fprintf('  --- Synaptic weights: %d units/cluster \n',NcUnits);
            Numbg=round(N_e*(1-f*p)); % number of background (i.e. non-selective) Exc units
            fprintf('  --- fraction of bg Exc units: %0.03g',Numbg/N_e);
            jee_in=jee_in*ones(1,Q);
            switch Network.clust
                case 'hom'
                    popsize=repmat(NcUnits,Q,1); % size of each Exc cluster
                case 'het'
                    Nc=[];
                    clust_std=Network.clust_std;
                    while (sum(Nc)-(N_e-Numbg))~=0 || any(Nc<0)
                        Nc=round(NcUnits+(NcUnits*clust_std)*randn(Q,1));
                    end
                    popsize=Nc; % array of cluster sizes
                    if any(sum(popsize)-(N_e-Numbg))
                        fprintf('\n---ERROR: Heterogeneous clusters: Problem with cluster numbers\n');
                    end
                    fprintf('\n  --- het cluster sizes->rescale Jplus in each cluster. JEE+:');
                    jee_in=jee_in*mean(popsize)./popsize';
                    disp(jee_in);
            end
            cusumNcE=[0 cumsum(popsize)'];
            % background units (if present), if not, override in next line
            JEE=(jee*(ones(N_e)+delta*randn(N_e,N_e))).*(rand([N_e,N_e])<peeout);
            JEI=(jei*(ones(N_e,N_i)+deltaEI*randn(N_e,N_i))).*(rand([N_e,N_i])<pei);
            JIE=(jie*(ones(N_i,N_e)+deltaIE*randn(N_i,N_e))).*(rand([N_i,N_e])<pie);
            JII=(jii*(ones(N_i)+delta*randn(N_i,N_i))).*(rand([N_i,N_i])<pii);
            if strcmp(Network.clust,'het') || strcmp(Network.clust,'hom')
                % clustered units: inter-cluster weights
                JEE(1:cusumNcE(Q+1),1:cusumNcE(Q+1))=...
                    (jee_out*(ones(cusumNcE(Q+1))+delta*randn(cusumNcE(Q+1),...
                    cusumNcE(Q+1)))).*(rand([cusumNcE(Q+1),cusumNcE(Q+1)])<peeout); % inter-cluster weights
                for clu=2:Q+1 % intra-cluster higher weights
                    JEE(1+cusumNcE(clu-1):cusumNcE(clu),1+cusumNcE(clu-1):cusumNcE(clu))=...
                        (jee_in(clu-1)*(ones(popsize(clu-1))+delta*randn(popsize(clu-1),popsize(clu-1)))).*...
                        (rand([popsize(clu-1),popsize(clu-1)])<peein);
                end
            end
            clustermatrix=eye(Q);
            
            if strcmp(Opt,'EI')
                % INHIBITORY CLUSTERS
                if any(strcmp(clusters,'EI'))
                    %     JminusEI = 1.-gam*fI*(JplusEI-1.);
                    JplusEI = 1/(1/p+(1-1/p)/factorEI);
                    JminusEI = JplusEI/factorEI;
                    params.JminusEI = JminusEI;
                    jei_out=-JminusEI*Jei; % intra-cluster
                    jei_in=-JplusEI*Jei; % inter-cluster
                    fprintf('JplusEI=%0.03g, JminusEI=%0.03g\n',JplusEI,JminusEI);
                end
                if any(strcmp(clusters,'IE'))
                    %     JminusIE = 1.-gam*fI*(JplusIE-1.);
                    JplusIE = 1/(1/p+(1-1/p)/factorIE);
                    JminusIE = JplusIE/factorIE;
                    params.JminusIE = JminusIE;
                    jie_out=JminusIE*Jie; % intra-cluster
                    jie_in=JplusIE*Jie; % inter-cluster
                    fprintf('JplusIE=%0.03g, JminusIE=%0.03g\n',JplusIE,JminusIE);
                end
                if any(strcmp(clusters,'II'))
                    JplusII=factorII;
                    JminusII = 1.-gam*fI*(JplusII-1.);
                    params.JminusII = JminusII;
                    jii_out=-JminusII*Jii; % intra-cluster
                    jii_in=-JplusII*Jii; % inter-cluster
                    fprintf('JplusII=%0.03g, JminusII=%0.03g\n',JplusII,JminusII);
                end
                %-------------
                % EI CLUSTERS
                %-------------
                % check #clusters and coding level are consistent
                if any([strcmp(clusters,'EI'), strcmp(clusters,'EI'), strcmp(clusters,'II')])
                    NcUnits=round(fI*N_i);    %  number of Exc units per cluster
                    fprintf('  --- Synaptic weights: %d units/cluster \n',NcUnits);
                    Numbg=round(N_i*(1-fI*p)); % number of background (i.e. non-selective) Exc units
                    fprintf('  --- fraction of bg Inh units: %0.03g',Numbg/N_i);
                    switch Network.clustEI
                        case 'hom'
                            popsizeI=repmat(NcUnits,Q,1);
                        case 'het'
                            Nc=[];
                            clust_std=Network.clust_std;
                            while (sum(Nc)-(N_i-Numbg))~=0 || any(Nc<0)
                                Nc=round(NcUnits+(NcUnits*clust_std)*randn(Q,1));
                            end
                            popsizeI=Nc; % array of cluster sizes
                            if any(sum(popsizeI)-(N_i-Numbg))
                                fprintf('\n---ERROR: Heterogeneous clusters: Problem with cluster numbers\n');
                            end
                    end
                    cusumNcI=[0 cumsum(popsizeI)'];
                    
                    %-------------
                    % EI weights
                    %-------------
                    if any(strcmp(Network.clusters,'EI'))
                        % background units (if present), if not, override in next line
                        if strcmp(Network.clustEI,'het') || strcmp(Network.clustEI,'hom')
                            % clustered units: inter-cluster weights
                            JEI(1:cusumNcE(Q+1),1:cusumNcI(Q+1))=...
                                (jei_out*(ones(cusumNcE(Q+1),cusumNcI(Q+1))+deltaEI*randn(cusumNcE(Q+1),...
                                cusumNcI(Q+1)))).*(rand([cusumNcE(Q+1),cusumNcI(Q+1)])<peiout); % inter-cluster weights
                            for clu=2:Q+1 % intra-cluster higher weights
                                JEI(1+cusumNcE(clu-1):cusumNcE(clu),1+cusumNcI(clu-1):cusumNcI(clu))=...
                                    (jei_in*(ones(popsize(clu-1),popsizeI(clu-1))+deltaEI*randn(popsize(clu-1),popsizeI(clu-1)))).*...
                                    (rand([popsize(clu-1),popsizeI(clu-1)])<peiin);
                            end
                        end
                    end
                    
                    %-------------
                    % IE weights
                    %-------------
                    if any(strcmp(Network.clusters,'IE'))
                        % background units (if present), if not, override in next line
                        if strcmp(Network.clustIE,'het') || strcmp(Network.clustIE,'hom')
                            % clustered units: inter-cluster weights
                            JIE(1:cusumNcI(Q+1),1:cusumNcE(Q+1))=...
                                (jie_out*(ones(cusumNcI(Q+1),cusumNcE(Q+1))+deltaIE*randn(cusumNcI(Q+1),...
                                cusumNcE(Q+1)))).*(rand([cusumNcI(Q+1),cusumNcE(Q+1)])<pieout); % inter-cluster weights
                            for clu=2:Q+1 % intra-cluster higher weights
                                JIE(1+cusumNcI(clu-1):cusumNcI(clu),1+cusumNcE(clu-1):cusumNcE(clu))=...
                                    (jie_in*(ones(popsizeI(clu-1),popsize(clu-1))+deltaIE*randn(popsizeI(clu-1),popsize(clu-1)))).*...
                                    (rand([popsizeI(clu-1),popsize(clu-1)])<piein);
                            end
                        end
                    end
                    
                    %-------------
                    % II weights
                    %-------------
                    if any(strcmp(Network.clusters,'II'))
                        % background units (if present), if not, override in next line
                        if strcmp(Network.clustII,'het') || strcmp(Network.clustII,'hom')
                            % clustered units: inter-cluster weights
                            JII(1:cusumNcI(Q+1),1:cusumNcI(Q+1))=...
                                (jii_out*(ones(cusumNcI(Q+1),cusumNcI(Q+1))+deltaII*randn(cusumNcI(Q+1),...
                                cusumNcI(Q+1)))).*(rand([cusumNcI(Q+1),cusumNcI(Q+1)])<piiout); % inter-cluster weights
                            for clu=2:Q+1 % intra-cluster higher weights
                                JII(1+cusumNcI(clu-1):cusumNcI(clu),1+cusumNcI(clu-1):cusumNcI(clu))=...
                                    (jii_in*(ones(popsizeI(clu-1),popsizeI(clu-1))+deltaII*randn(popsizeI(clu-1),popsizeI(clu-1)))).*...
                                    (rand([popsizeI(clu-1),popsizeI(clu-1)])<piiin);
                            end
                        end
                    end
                    params.popsizeI=popsizeI;
                end
            end
            
            JEI(JEI>0)=0;
            JIE(JIE<0)=0;
            JII(JII>0)=0;
            JEE(JEE<0)=0;
            J=[JEE JEI; JIE JII];
            J=J-diag(diag(J)); % eliminate self-couplings
            fprintf('  --- New synaptic weights set: done...\n');
            fprintf('      Overall: Jee=%g -- Jie=%g -- Jei=%g -- Jii=%g \n',jee,jie,jei,jii);
            fprintf('      Var[J]=(Jx%0.03g)^2\n',delta);
            
            params.popsize=popsize;
            params.clustermatrix=clustermatrix;
            
            if strcmp(Network.clust,'het')
                for i=1:Q
                    a(i)=sum(popsize(clustermatrix(:,i)>0));
                end
                fprintf('  --- clusters size: mean=%0.03g neurons/cluster, sd/mean=%0.03g\n',mean(a),std(a)/mean(a));
            end
            fprintf('  --- fraction of bg Exc units: %0.03g\n',N_e/Numbg);
            
            %-------------------
            % PLOT weight matrix
            %-------------------
            figure(1); clf;
            subplot(2,1,1);
            colormap(utils.redblue); %xy=J; fun_colormapLim;
            imagesc(J);
            utils.figset(gca,'neurons','neurons','weights',10);
            colorbar;
            subplot(2,1,2);
            lambda=eig(J);
            plot(real(lambda),imag(lambda),'.');
            utils.figset(gca,'Re(\lambda)','Im(\lambda)','eig(weights)',10);
            saveas(gcf,fullfile(params.savedir,'weights.pdf'),'pdf');
            
            % spike thresholds for each population
            theta=[params.theta_e, params.theta_i];
            Theta=zeros(1,numel(params.popsize)+2);
            Theta(1:numel(params.popsize)+1)=theta(1);
            Theta(end) = theta(2);
            params.Theta=Theta;
            fprintf('Spike thresholds Theta calculated for each population and stored in params\n');
        end
        
        %% fun_stim generate stimulus profile and selectivity indices
        
        function [stimulus_save, params]=fun_stim(params)
            
            % unpack vars
            p=params.p;
            Stimulus=params.Stimulus;
            stimuli=params.stimuli;
            Sim=params.Sim;
            
            % for each event, create external current and  stim properties in .Ext
            % and add clusters selective to the event .Stimulus.feat(n).StimClust
            stimulus_save=struct('Ext',[],'Stimulus',[]);
            
            % select stimuli
            temp_Stimulus=struct('input',Stimulus.input);
            indfeat=zeros(1,numel(stimuli));
            
            for ev=1:numel(stimuli)
                fprintf('Stimulus %s',stimuli{ev});
                % match current stimuli to features in Stimulus
                indfeat(ev)=find(cell2mat(arrayfun(@(x)strcmp(stimuli{ev},x(:).name),...
                    Stimulus.feat(:),'uniformoutput',false)));
            end
            fprintf('\n');
            if ~isempty(indfeat)
                temp_Stimulus.feat(1:numel(indfeat))=Stimulus.feat(indfeat);
                
                % define stimulus selectivity: which clusters each stimulus
                % targets
                for n=1:numel(indfeat)
                    sclust=[];
                    
                    switch Stimulus.feat(n).selectivity
                        case {'mixed','allsel'}
                            % this is the option for the US stimulus
                            pstim=0.5; % probability that a cluster is selective to a stimulus
                            selective=rand(1,p)<pstim;
                        case 'exc'
                            % this is the option for the perturbation: CSgauss stimulus (all neurons
                            % are selective)
                            selective=ones(1,p);
                    end
                    temp_Stimulus.feat(n).selective=selective;
                    
                    
                    if ~isempty(temp_Stimulus.feat(n).selective)
                        sclust=find(temp_Stimulus.feat(n).selective(1,:));
                    end
                    temp_Stimulus.feat(n).StimClust=sclust;
                end
            end
            Stimulus=temp_Stimulus;
            Ext=struct('Mu',[]);
            
            % LOAD PARAMETERS
            fieldNames={'Sim','Network','p','popsize','clustermatrix','N_e','N_i','Cext','Jee_ext','Jie_ext','ni_ext','tau_e','tau_i','fieldNames'};
            utils.v2struct(params,fieldNames);
            cusumNcE=[0 cumsum(popsize)'];
            Tseq=Sim.t_Start:Sim.dt_step:Sim.t_End;
            
            if ~isempty(stimuli)
                feat=Stimulus.feat;
                nstim=numel(feat); % number of stimuli in current trials
                stim=repmat(struct('profile',[],'ind',[],'interval',[]),1,nstim);
                temp_ind=repmat(struct('ind',[]),1,nstim); % stores indices for mixed cue (see below)
                for n=1:nstim
                    % stimulus interval
                    interv=feat(n).interval;
                    Gain=feat(n).gain;
                    if ~isempty(strfind(feat(n).name,'gauss'))
                        Gain=1; % with gaussian stim set profile to peak at 1, then multiply each profile by gaussian with SD feat(n).gain for each neuron in feat(n).gauss
                    end
                    Profile=feat(n).profile;
                    Profile=@(t)Profile(t-interv(1));
                    MaxThInput=max(abs(Profile(Tseq(Tseq>interv(1) & Tseq<interv(2)))));
                    Profile=@(t)Gain*Profile(t)/MaxThInput;
                    stim(n).profile=@(t)Profile(t); % fraction increase above baseline
                    % selective neurons
                    StimClust=Stimulus.feat(n).StimClust; % clusters activated by current stimulus
                    % units selective to stimulus
                    ind=[]; % indices of stim sel units
                    switch feat(n).selectivity
                        case 'mixed'
                            for c=StimClust
                                pop_ind=find(clustermatrix(:,c));
                                for k=1:numel(pop_ind)
                                    ind=[ind cusumNcE(pop_ind(k))+1:cusumNcE(pop_ind(k)+1)]; % stim selective units
                                end
                            end
                        case 'exc'
                            ind=1:N_e;
                        otherwise
                            ind=1:N_e;
                    end
                    % sparsify
                    a=randperm(numel(ind));
                    temp_ind(n).ind=ind;
                    ind=ind(a(1:round(feat(n).connectivity*numel(ind))));
                    % gaussian stimulus, draw from randn
                    if ~isempty(strfind(feat(n).name,'gauss'))
                        stim(n).gauss=feat(n).gain*randn(numel(ind),1);
                    end
                    %
                    stim(n).ind=ind;
                    stim(n).interval=interv;
                    stim(n).name=feat(n).name;
                    stim(n).StimClust=StimClust;
                    stim(n).selectivity=feat(n).selectivity;
                    
                end
                Ext.stim=stim;
            end
            Ext.Mu=params.Mu;
            stimulus_save.Ext=Ext;
            stimulus_save.Stimulus=temp_Stimulus;
            
        end
        
        
        %%
        % SIM of one trials given parameters
        %
        % Luca Mazzucato March 2014
        
        % SET OPTIONS
        % ParamsRun = structure containing parameters for simulation
        
        
        function [all_firings, PlotData]=fun_LIF_SIM(ParamsRun)
            
            
            Theta=ParamsRun.Theta; Sim=ParamsRun.Sim; stimuli=ParamsRun.stimuli;%Stimulus=ParamsRun.Stimulus;
            Ext=ParamsRun.Ext; J=ParamsRun.J; N_e=ParamsRun.N_e; N_i=ParamsRun.N_i; p=ParamsRun.p; He=ParamsRun.He;
            Hi=ParamsRun.Hi; tau_e=ParamsRun.tau_e; tau_i=ParamsRun.tau_i; tausyn_e=ParamsRun.tausyn_e;
            tausyn_i=ParamsRun.tausyn_i; tau_arp=ParamsRun.tau_arp;
            %
            all_firings=[];
            dt=Sim.dt_step;            % time step (s)
            Tseq=Sim.t_Start:dt:Sim.t_End;
            
            %--------------------
            % PARAMETERS
            %--------------------
            % CELL
            VEreset=He*Theta(1);
            VIreset=Hi*Theta(end);
            %
            %----------
            % STIMULUS
            %----------
            % add stimuli on top of baseline: for each stimulus provide
            %              - profile (perc. increase on baseline current)
            %              - index of selective neurons
            % BASELINE EXTERNAL CURRENT
            mu=Ext.Mu; % mu is an (N_e+N_i)-th array
            if ~isempty(stimuli)
                stim=Ext.stim;
                nstim=numel(stim); % number of stimuli in current trials
            end
            %----------------
            % SYNAPTIC FILTER
            %----------------
            Tau.tausyn_e=tausyn_e; % exc synaptic time (fall time)
            Tau.tausyn_i=tausyn_i; % exc synaptic time (fall time)
            F=synaptic_trace(Tau,dt,N_e,N_i); % traces for recurrent connections
            %--------------------------
            % SIMULATION
            %--------------------------
            % preallocate memory for stored variable firings_tmp
            % INITIAL CONDITIONS: random
            v=[(Theta(1)-VEreset)/2*ones(N_e,1)+(Theta(1)-VEreset)/2*(2*rand(N_e,1)-1);...
                (Theta(end)-VIreset)/2*ones(N_i,1)+(Theta(end)-VIreset)/2*(2*rand(N_i,1)-1)];  % Initial values of v
            % THRESHOLD VECTOR
            VTh=[Theta(1)*ones(N_e,1); Theta(end)*ones(N_i,1)];
            c=[VEreset*ones(N_e,1);  VIreset*ones(N_i,1)]; % reset potentials
            % fprintf('\nVEThres=%g --- VIThres=%g',VTh(1),VTh(end));
            % fprintf('\nVEreset=%g --- VIreset=%g \n',c(1),c(end));
            % Excitatory neurons        Inhibitory neurons
            tau=[tau_e*ones(N_e,1);       tau_i*ones(N_i,1)];
            %
            firings=zeros(10*numel(Tseq),2);
            firings_cnt=0;
            tic
            %--------------------
            % PLOT
            %--------------------
            PlotData=[];
            PlotData.Ne_plot=N_e; % number of exc neuron to plot
            PlotData.Ni_plot=N_i; % number of inh neurons to plot
            ind_plot=[5; N_e+5]; % indices of neurons to plot
            if ~isempty(stimuli)
                indcue=find(cellfun(@(x)~isempty(x),strfind({stim(:).name},'CS')));
                if ~isempty(indcue)
                    ind_plot(1)=stim(indcue).ind(1);
                end
            end
            nplot=numel(ind_plot); % number of neurons to plot (membrane potential plot)
            vi=0; % running index for vplot
            PlotData.vplot = zeros(nplot,round(Sim.plot_length/dt)); % store membrane potential for plots; rows=neurons, cols=time steps;
            PlotData.iEplot = zeros(2,round(Sim.plot_length/dt)); % store EPSC for plots; rows=neurons, cols=time steps;
            PlotData.iExtplot = zeros(2,round(Sim.plot_length/dt)); % store IPSC for plots; rows=neurons, cols=time steps;
            PlotData.iIplot = zeros(2,round(Sim.plot_length/dt)); % store IPSC for plots; rows=neurons, cols=time steps;
            PlotData.p=p;
            PlotData.VTh=VTh;
            PlotData.tau=tau;
            PlotData.ind_plot=ind_plot;
            %----------------------------
            % RUN
            %----------------------------
            refr=zeros(size(mu,1),1);       % neurons in refractory state
            for t=1:numel(Tseq)         % siMulation of 1000 ms
                fired=find(v>VTh); % indices of spikes
                Isyn=zeros(N_e+N_i,1);
                % spikes
                if ~isempty(fired)
                    v(fired)=c(fired);
                    refr(fired)=tau_arp;
                end
                % recurrent synaptic current
                F=syn_evolve(F,fired);
                % integrate
                muRun=mu;
                if ~isempty(stimuli)
                    for n=1:nstim
                        if Tseq(t)>=stim(n).interval(1) && Tseq(t)<=stim(n).interval(2)
                            if strcmp(stim(n).name,'CSgauss')
                                muRun(stim(n).ind)=muRun(stim(n).ind)+stim(n).profile(Tseq(t))*mu(stim(n).ind).*stim(n).gauss;
                            else
                                muRun(stim(n).ind)=muRun(stim(n).ind)+stim(n).profile(Tseq(t))*mu(stim(n).ind);
                            end
                        end
                    end
                end
                Isyn=Isyn+J*F.f;
                v=v-v*dt./tau+muRun(:)*dt+Isyn*dt;
                % neurons in refractory state
                refr=max(-0.001,refr-dt);
                v(refr>0)=c(refr>0);
                % store spikes
                if ~isempty(fired)
                    % if firings_tmp has no more space, preallocate more memory
                    if firings_cnt+numel(fired)>size(firings,1)
                        firings=[firings; zeros(10*numel(Tseq),2)];
                    end
                    firings(firings_cnt+1:firings_cnt+numel(fired),1:2)=[Tseq(t)+0*fired, fired];
                    firings_cnt=firings_cnt+numel(fired);
                end
                % store values for plotting, only last Sim.plot_length interval
                if Tseq(t)>Sim.t_End-Sim.plot_length
                    vi=vi+1;
                    % membrane potential
                    PlotData.vplot(1:nplot,vi)=v(ind_plot);
                    % input currents
                    PlotData.iEplot(1:nplot,vi)=J(ind_plot,1:N_e)*F.f(1:N_e);
                    PlotData.iIplot(1:nplot,vi)=J(ind_plot,N_e+1:N_e+N_i)*F.f(N_e+1:N_e+N_i);
                    PlotData.iExtplot(1:nplot,vi)=muRun(ind_plot,1);
                end
            end
            fprintf('--- End of trial...\n');
            toc
            %---------------------------------------------------------------------------
            if ~any(any(firings))
                fprintf('\n --- NO SPIKES GENERATED... \n');
            else
                % find last spike in firings
                IndexEnd=find(firings(:,2)==0,1)-1;
                if isempty(IndexEnd)
                    IndexEnd=size(firings,1);
                end
                all_firings=firings(1:IndexEnd,[1 2]);
            end
            
            
            function F=synaptic_trace(Tau,dt,N_e,N_i)
                
                F=struct();
                tau_sE=Tau.tausyn_e; % exc synaptic time (fall time)
                tau_sI=Tau.tausyn_i; % inh synaptic time (fall time)
                fexp=[repmat(exp(-dt/tau_sE),N_e,1); repmat(exp(-dt/tau_sI),N_i,1)]; % Multiplicative step (fp)
                fSpike=[repmat((1/tau_sE),N_e,1); repmat((1/tau_sI),N_i,1)]; % add to fp with a spike
                f=zeros(N_e+N_i,1);
                F.fexp=fexp;
                F.fSpike=fSpike;
                F.f=f;
                
            end
            
            function F=syn_evolve(F,fired)
                
                
                % update synaptic filter
                F.f=F.fexp.*F.f;
                if ~isempty(fired)
                    % update of synaptic filter with spikes
                    F.f(fired)=F.f(fired)+F.fSpike(fired);
                end
            end
            
            
        end
        
        
        
        
    end
end