#
from transformer_lens import EasyTransformer, EasyTransformerConfig
import tqdm.auto as tqdm

#
from tqdm import trange
from scipy.stats import beta
#
import torch
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"
import scipy, os
import numpy as np

from Visualize_dice_roll import Visualize_dice_roll


###############################################
################## CLASSES ####################
###############################################

class BayesianTransformer():
    
    def __init__(self, n_layers,
                       n_training_epochs,
                       n_episodes_batch,
                       n_trials,
                       game_type,
                       params,
                       dice_roll_params,
                       len_episode=10,
                       random_seed=False):
        #
        self.n_trials = n_trials

        #
        self.root_dir = '/media/cat/4TBSSD/btrans/'

        #
        if random_seed==False:
            torch.manual_seed(42)
            np.random.seed(seed=42)
        else:
            pass

        #
        self.len_episode = len_episode
        
        #
        self.game_type = game_type

        #
        self.coin_toss_params = params
        print ('... coin toss params: ', self.coin_toss_params)
        # old way of specifying the params
        self.a0 = params[0]
        self.b0 = params[1]

        #
        self.dice_roll_params = dice_roll_params
        
        #
        self.n_layers = n_layers
        
        #
        self.n_training_epochs = n_training_epochs
        
        #
        self.n_episodes_batch = n_episodes_batch

        # 
        self.params = self.dice_roll_params if self.game_type == "dice_roll" else self.coin_toss_params
        
        print ("... params: ", self.params)
    
    # make model for dice roll
    def make_model_dice_roll(self):
        fname_model = os.path.join(self.root_dir,
                                    'model_nlayers_'+str(self.n_layers)
                                    +"_ntraining_"+str(self.n_training_epochs)
                                    +"_dice_roll_"
                                    +str(self.params)+".btrans").replace(',','')
        self.fname_model = fname_model                                   


    #
    def make_model_name(self):
       
        if self.use_multi_distributions:
            
             self.fname_model = os.path.join(self.root_dir,
                                                'model_nlayers_'+str(self.n_layers)
                                                +"_ntraining_"+str(self.n_training_epochs)
                                                +"_ncoin_tosses_"+str(self.n_trials)
                                                +"_coin_toss_multi_"
                                                +str(self.params)+".btrans")
        else:

            #
            fname_model_new2 = os.path.join(self.root_dir,
                                                'model_nlayers_'+str(self.n_layers)
                                                +"_ntraining_"+str(self.n_training_epochs)
                                                +"_ncoin_tosses_"+str(self.n_trials)
                                                +"_coin_toss_"
                                                +str(self.params)+".btrans")
                
            fname_model_new = os.path.join(self.root_dir,
                                                'model_nlayers_'+str(self.n_layers)
                                                +"_ntraining_"+str(self.n_training_epochs)
                                                +"_coin_toss_"
                                                +str(self.params)+".btrans")
            
            fname_model_old = os.path.join(self.root_dir,
                                        'model_nlayers_'+str(self.n_layers)
                                        +"_ntraining_"+str(self.n_training_epochs)
                                        +"_a0_"+str(self.params[0])
                                        +"_b0_"+str(self.params[1])+".btrans")

            print ("searching for model: ", fname_model_new2)

            # check if latest version of model exists
            if os.path.exists(fname_model_new2)==True:
                self.fname_model = fname_model_new2
            elif os.path.exists(fname_model_new)==True:
                self.fname_model = fname_model_new
            else:
                self.fname_model = fname_model_old

        # #
        # if game_type == "coin_toss":
        #     if bt.use_multi_distributions:
        #         bt.fname_model = os.path.join(bt.root_dir,
        #                             'model_nlayers_'+str(n_layers)
        #                             +"_ntraining_"+str(n_training_epochs)
        #                             +"_ncoin_tosses_"+str(n_trials)
        #                             +"_coin_toss_multi_"
        #                             +str(params_list)+".btrans")
        #     else:
        #         bt.fname_model = os.path.join(bt.root_dir,
        #                             'model_nlayers_'+str(n_layers)
        #                             +"_ntraining_"+str(n_training_epochs)
        #                             +"_ncoin_tosses_"+str(n_trials)
        #                             +"_coin_toss_"
        #                             +str(params)+".btrans")
                
        #         if os.path.exists(bt.fname_model)==False:

        #             print ("using old model name")

        #             bt.fname_model = os.path.join(bt.root_dir,
        #                             'model_nlayers_'+str(n_layers)
        #                             +"_ntraining_"+str(n_training_epochs)
        #                             +"_coin_toss_"
        #                             +str(params)+".btrans")

    


    #
    def init_model(self):

        #
        cfg = EasyTransformerConfig(
            n_layers = self.n_layers,
            d_model=64,
            d_head=64,
            n_heads=1,
            d_mlp=256,
            d_vocab=len(self.params)+1,               # vocabuliary length; heads/tails options
            n_ctx=self.n_trials+1,
            act_fn='relu',
            normalization_type="LN",
        )
        #
        self.model = EasyTransformer(cfg).to(device)
        #self.model = EasyTransformer(cfg)


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        #print ("... model initialized...")

    #    
    def make_data_generator_coin_toss_multi(self, 
                                            batch_size=4):

        # unpack the distributions
        
        x = np.linspace(0, 1, 1000)
        pdfs = []
        for k in range(len(self.params_list)):
            param = self.params_list[k]
            pdf = beta.pdf(x, param[0], param[1])
            pdfs.append(pdf)

        pdfs = np.array(pdfs)
        pdf_sum = np.sum(pdfs,0)

        # Normalize the PDF values to create a PMF
        pmf_sum = pdf_sum / np.sum(pdf_sum)

        #
        biases = []  
        coin_toss_array = []
        # TODO make this simpler; draw 1 time not N times
        for _ in range(batch_size):
          bias = np.random.choice(x, size=1, p=pmf_sum)
          #bias = distribution.rvs(size = 1)
          biases.append(bias)

          # generate a time series based on the bias now
          samples = np.random.rand(self.n_trials)

          # 
          coin_tosses = samples >= bias
          coin_tosses = coin_tosses + 0.0
          coin_tosses_with_bos = np.insert(coin_tosses, 0, 2., axis=0)
          coin_tosses_with_bos = torch.LongTensor(coin_tosses_with_bos)
          coin_toss_array.append(coin_tosses_with_bos)

        return torch.vstack(coin_toss_array).to(device), biases

    #
    def make_data_generator_coin_toss(self, batch_size=4, a0=1, b0=1):
        
        #
        distribution = scipy.stats.beta(a0,b0)

        biases = []  
        coin_toss_array = []
        # TODO make this simpler; draw 1 time not N times
        for k in range(batch_size):
          bias = distribution.rvs(size = 1)
          biases.append(bias)

          # generate a time series based on the bias now
          samples = np.random.rand(self.n_trials)

          # 
          coin_tosses = samples >= bias
          coin_tosses = coin_tosses + 0.0
          coin_tosses_with_bos = np.insert(coin_tosses, 0, 2., axis=0)
          coin_tosses_with_bos = torch.LongTensor(coin_tosses_with_bos)
          coin_toss_array.append(coin_tosses_with_bos)

        return torch.vstack(coin_toss_array).to(device), biases

    #
    def make_data_generator_dice_roll(self, batch_size=4):
        
        # we take the true params here
        theta_true = self.dice_roll_params

        # 
        dice_roll_array = []
        biases = []
        for k in range(batch_size):
          
          # so we first use the dirichlet distribution to draw a sample
          bias = np.random.dirichlet(theta_true)
          biases.append(bias)

          # and only then generate the dice rolls based on the sample for that episode
          dice_rolls = np.random.choice([0,1,2], size=self.len_episode, 
                                        p=bias)

          # note bos is 3 for a 3-sided dice
          dice_rolls_with_bos = np.insert(dice_rolls, 0, 3., axis=0)

          # convert to tensor
          dice_rolls_with_bos = torch.LongTensor(dice_rolls_with_bos)
          dice_roll_array.append(dice_rolls_with_bos)

        return torch.vstack(dice_roll_array).to(device), biases

    #
    def run_model_dice_roll(self):
    
        #
        self.losses = []
        self.biases = []
        for epoch in tqdm.tqdm(range(self.n_training_epochs),
                               desc="params: "+str(self.a0)+" "+str(self.b0)+
                               " "+str(self.n_training_epochs)+
                               " "+str(self.n_layers)):
            #
            tokens, biases = self.make_data_generator_dice_roll(self.n_episodes_batch)
            self.biases.append(biases)
            
            #
            loss = self.model(tokens, return_type="loss")
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.losses.append(loss.cpu().detach().numpy())
            #if epoch % 10 == 0:
            #    print(f"Epoch: {epoch}. Loss: {loss}")

        if self.plotting:
            plt.figure()
            plt.plot(self.losses)
            plt.show()
        #print ("...done training...")

    #
    def run_model_coin_toss(self):
    
        #
        self.losses = []
        for epoch in tqdm.tqdm(range(self.n_training_epochs),
                               desc="params: "+str(self.a0)+" "+str(self.b0)+
                               " "+str(self.n_training_epochs)+
                               " "+str(self.n_layers)):
            #
            if self.use_multi_distributions:
                tokens,_ = self.make_data_generator_coin_toss_multi(self.n_episodes_batch)
    
            else:
                tokens,_ = self.make_data_generator_coin_toss(self.n_episodes_batch, 
                                                          self.a0, 
                                                          self.b0)

            #
            loss = self.model(tokens, return_type="loss")
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.losses.append(loss.cpu().detach().numpy())

            # track the accuracy by predicting every 10th epoch
            if self.predict_during_training:
                if epoch % self.predict_during_training_interval == 0:
                    vis = VisualizeBT_coin_toss(self)
                    vis.n_trials = self.n_trials
                    vis.bt = self
                    vis.generate_examples_coin_toss(self.n_test_trials)
                    
                    #
                    vis.save_run_coin_toss(self.fname_model[:-7]+"_epoch"
                                           +str(epoch)+".npz")


            #if epoch % 100 == 0:
            #    print(f"Epoch: {epoch}. Loss: {loss}")

        #print ("...done training...")

    #
    def save_model(self):

        #
        torch.save(self.model, self.fname_model)


    #
    def load_model(self):

        # reset cuda device
        #!CUDA_LAUNCH_BLOCKING=1
        print ("... resetting cuda device ...")
        print ("... cuda available: ", torch.cuda.is_available())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        #
        self.init_model()

        #
        print ("Loading: ", self.fname_model)
        self.model = torch.load(self.fname_model)
        self.model.to('cpu')

        #print ("... model loaded ...")
        #print ("self.model: ", self.model)

        #
        #print ("... model loaded ...")



#
class VisualizeBT_coin_toss():

    #
    def __init__(self, bt=None):

        #
        self.bt = bt

        #
        self.root_dir = '/media/cat/4TBSSD/btrans/'

        #

    #
    def save_run_coin_toss(self, fname_out):

        # see if loss variable exists; when we load model from scratch it doesn't remember losses
        if not hasattr(self.bt, 'losses'):
            print ("losses not found; exiting")
            self.bt.losses = None

        #
        np.savez(fname_out,
                 kls_bayesian = self.kls_bayesian,
                 kls_frequentist = self.kls_frequentist,
                 trans_pred = self.trans_pred,
                 bayes_pred = self.bayes_pred,
                 freq_pred = self.freq_pred,
                 biases = self.biases,
                 losses = self.bt.losses
                )
#
    def generate_examples_coin_toss(self, n_trials):
        import scipy
        
        #
        biases = []
        kls_bayesian = []
        kls_frequentist = []
        trans_pred = []
        bayes_pred = []
        freq_pred = []

        # TODO: simplify this to do all batches in 1 step...
        for ctr,test_id in enumerate(trange(n_trials, desc='testing')):

              test_data, bias = self.bt.make_data_generator_coin_toss(1, self.bt.a0, self.bt.b0)
              biases.append(bias)
              #######################################################
              ############### TRANSFORMER RESULTS ###################
              #######################################################
              test_data = test_data[0]
              test_data_nobos = test_data[1:]
              logits = self.bt.model(test_data)

              #
              if False:
                logits = logits.cpu().detach().numpy()[0]
                transformer_prediction = []
                for logit in logits:
                    temp = np.exp(logit)/sum(np.exp(logit))
                    temp = temp[0]/np.sum(temp)
                    transformer_prediction.append(temp)

                trans_pred.append(transformer_prediction)
              else:
                  #print ("using new method...")
                  #trans_pred = torch.nn.functional.softmax(logits, dim=1)[:,:3]
                  #trans_pred = trans_pred.cpu().detach().numpy()
                  trans_pred = torch.nn.functional.softmax(logits[0], dim=1)[:,0]
                  trans_pred = trans_pred.squeeze().cpu().detach().numpy()
        


              #######################################################
              ################# PLOT BAYESIAN OPTIMAL RESULTS #######
              #######################################################
              # update prior using bernoulli update rule inference
              bayesian = []
              confs = []
            #   x = np.linspace(0.00,1, 100)
            #   x_plot = np.arange(0,self.n_trials,1)

              # generate first value
              dist = scipy.stats.beta(self.bt.a0,self.bt.b0)
              bayesian.append(dist.mean())

              interval = dist.interval(0.95)
              confs.append(interval)

              #
              a=self.bt.a0
              b=self.bt.b0
              for q in range(0,test_data_nobos.shape[0],1):
                outcome = test_data_nobos[q]#.numpy()
                #print ("bayesian outcome: ", outcome)
                if outcome == 0:
                    a+=1
                else:
                    b+=1

                #
                dist = scipy.stats.beta(a,b)
                bayesian.append(dist.mean())

                interval = dist.interval(0.95)
                confs.append(interval)

              #
              bayesian = np.array(bayesian)
              bayes_pred.append(bayesian)

              #######################################################
              ################# PLOT FREQUENTIST RESULTS ############
              #######################################################
              # update prior using bernoulli update rule inference
              x_plot = np.arange(1,self.bt.n_trials+1,1)

              #
              frequentist = []
              for q in range(0,test_data_nobos.shape[0],1):
                outcome = test_data_nobos[:q+1]

                idx = np.where(outcome.cpu().numpy()==1)[0]
                frequentist.append(1-idx.shape[0]/len(outcome))

              freq_pred.append(frequentist)

              ###############################################################
              ####################### KL ENTROPY ############################
              ###############################################################

              kls = [] 
              for k in range(len(trans_pred)):
                temp_b = bayesian[k] #+0.00001
                temp_t = trans_pred[k] #+0.00001
                #res = scipy.special.rel_entr(temp_t, temp_b)
                res = scipy.special.kl_div(temp_t, temp_b)
                
                #print (temp_b, temp_t, res)
                kls.append(res)

              kls_bayesian.append(kls)

              #
              kls = [] 
              for k in range(1,len(trans_pred),1):
                temp_t = trans_pred[k]#+0.00001
                temp_f = frequentist[k-1]#+0.00001
                res = scipy.special.kl_div(temp_t, temp_f)
                kls.append(res)
              kls_frequentist.append(kls)

        #
        self.kls_bayesian = kls_bayesian
        self.kls_frequentist = kls_frequentist
        self.trans_pred = trans_pred
        self.bayes_pred = bayes_pred
        self.freq_pred = freq_pred
        self.biases = biases

    #
    def load_run(self, fname_in):

        d = np.load(fname_in)

        self.kls_bayesian = d['kls_bayesian']
        self.kls_frequentist = d['kls_frequentist']
        self.trans_pred = d['trans_pred']
        self.bayes_pred = d['bayes_pred']
        self.freq_pred = d['freq_pred']
        self.biases = d['biases']
        self.losses = d['losses']

    #
    def show_examples(self, 
                      n_examples, 
                      examples=None, 
                      biases=None):
		
        #
        import scipy

        #
        fig1 = plt.figure(figsize=(5,4), facecolor='white')
        ax1 = fig1.add_subplot(1,1,1)
        ax1.set_ylim([-0.2,1])

        #
        plt.suptitle("Beta("+str(self.bt.a0)+","+str(self.bt.b0)+")")
        
        #
        if self.show_kl_entropy:
            fig2 = plt.figure(figsize=(5,4), facecolor='white')
        
        #
        if examples is None:
            examples = np.arange(n_examples)
            randomize_examples = True
        else:
            randomize_examples = False


        for k in range(n_examples):

          #
          if randomize_examples:
              test_data, bias = self.bt.make_data_generator_coin_toss(1, 
                                                                  self.bt.a0, 
                                                                  self.bt.b0)
              test_data = test_data[0]
              test_data_cpu = test_data.cpu().detach().numpy()
              test_data_nobos = test_data[1:].cpu().detach().numpy()
          else:
              test_data = examples
              test_data = torch.LongTensor(test_data)
              test_data.to('cuda')

              # 
              test_data_cpu = test_data.cpu().detach().numpy()
              test_data_nobos = test_data[1:].cpu().detach().numpy()
              
              #
              bias = biases
          print ("test data: ", test_data)


          #######################################################
          ########## LOAD TRANSFORMER RESULTS ###################
          #######################################################
          logits = self.bt.model(test_data)

          #
          logits = torch.nn.functional.softmax(logits[0], dim=1)[:,0]
          transformer_prediction = logits.squeeze().cpu().detach().numpy()
          print ("transformer prediction: ", transformer_prediction)
          x_plot = np.arange(0,self.bt.n_trials+1,1)
          ax1.plot(x_plot, transformer_prediction, c= 'steelblue', label='transformer prediction')
          ax1.scatter(x_plot, transformer_prediction, c= 'steelblue')

          #######################################################
          ################# PLOT BAYESIAN OPTIMAL RESULTS #######
          #######################################################
          
          # if we're not doing multi-distribution, then we can compute the optimal bayesian analytically:
          if self.bt.use_multi_distributions==False:
                
            dist = scipy.stats.beta(self.bt.a0,self.bt.b0)
            # 
            bayesian = []
            confs = []

            y = dist.mean()
            bayesian.append(y)

            interval = dist.interval(0.95)
            confs.append(interval)

            #
            a=self.bt.a0
            b=self.bt.b0
            for q in range(0,test_data_nobos.shape[0],1):
                outcome = test_data_nobos[q]#.numpy()
                #print ("bayesian outcome: ", outcome)
                if outcome == 0:
                    a+=1
                else:
                    b+=1

                #
                dist = scipy.stats.beta(a,b)
                y = dist.mean()
                bayesian.append(y)

                # show all these params by printing
                print ("a: ", a, " b: ", b, " y: ", y)
                

                interval = dist.interval(0.95)
                confs.append(interval)

                #
            bayesian = np.array(bayesian)
            print ("bayesian optimal: ", bayesian)
            ax1.plot(x_plot,bayesian, c='green',label='bayesian (shading 95% conf int)')
            ax1.scatter(x_plot,bayesian, c='green')


          else:
            
                print ("loading bayesian optimal from pymc results")
                fname = os.path.join('/media/cat/4TBSSD/btrans/'
                                + "pymc3_"+str(self.bt.coin_toss_params)+"_observed_bias_"+
                                str(biases[0])+'.npz')
                print ("loading: ", fname)
                try:
                    print ("loading: ", fname)
                    data = np.load(fname, allow_pickle=True)
                    traces = data['traces']
                
                    #
                    bayes_mean = []
                    bayes_median = []
                    for k in range(len(traces)):
                        trace = traces[k]
                        temp_ = []
                        for q in range(len(trace)):
                            temp_.append(trace[q]['prior'])

                        #
                        bayes_mean.append(np.mean(temp_))
                        bayes_median.append(np.median(temp_))

                    #
                    observed_bias = data['observed_bias']
                    observed_data_all = data['observed_data_all']

                    #
                    print ("bayes mean: ", bayes_mean)
                    ax1.plot(x_plot,bayes_mean, c='green',label='pymc3: bayesian mean')
                    ax1.scatter(x_plot,bayes_mean, c='green')
                    ax1.plot(x_plot,bayes_median, c='magenta',label='pymc3: bayesian median')
                    ax1.scatter(x_plot,bayes_median, c='magenta')
                except:
                    print ("missing pymc3 data")
                    

          #confs=np.vstack(confs)

          #
          if False:
              ax1.fill_between(x_plot, bayesian - confs[:,0], bayesian + confs[:,-1],
                        color='green', alpha=.1)

          #####################################################
          ################ COIN TOSS PLOTS ####################
          #####################################################
          # heads
          x = np.where(test_data_cpu==0)[0]-0.5
          ax1.vlines(x, ymin=x*0-.2, ymax=x*0-.12, colors='blue', lw=2, label='heads')

          # tails
          x = np.where(test_data_cpu==1)[0]-0.5
          ax1.vlines(x, ymin=x*0-.1, ymax=x*0-0.02, colors='red', lw=2, label='tails')

          # plot drawn coin bias
          #axtwinx = ax1.twinx()
          #axtwinx.set_ylim([-0.2,1])
          xx = x_plot
          yy = np.zeros(x_plot.shape[0])+bias
          yy = yy.squeeze()
          #print ("xx: ", xx)
          #print ("yy: ", yy)
          ax1.fill_between(x_plot, yy-0.025, yy+0.025,
                                color='black', alpha=.2)
          #axtwinx.scatter(10,bias[0],c='black', s=100, label='coin bias')

          #######################################################
          ################# PLOT FREQUENTIST RESULTS ############
          #######################################################
          # update prior using bernoulli update rule inference
          if self.show_frequentist:
            x_plot = np.arange(1,self.bt.n_trials+1,1)

            #
            frequentist = []
            for q in range(0,test_data_nobos.shape[0],1):
                outcome = test_data_nobos[:q+1]

                idx = np.where(outcome==1)[0]
                frequentist.append(1-idx.shape[0]/len(outcome))

            ax1.plot(x_plot, frequentist, c='pink',label='frequentist')
            ax1.scatter(x_plot, frequentist, c='pink')

          #########################################################
          #if ctr==0:
          ax1.legend(loc=1,fontsize=8)
          ax1.set_xlabel("prediction # (integers) / toss # (half-integers)")
          ax1.set_ylabel("Prob of heads")

          #ax1.set_ylim(-0.2,1.1)
          #print ("bias: ", bias)
          #print ("bias shape: ", bias.shape)
          try:
              ax1.set_title("Coin bias: "+str(round(bias[0],4)))
          except: 
              ax1.set_title("Coin bias: "+str(round(bias[0][0],4)))

          ###############################################################
          ####################### KL ENTROPY ############################
          ###############################################################
          import scipy
          if self.show_kl_entropy:
            ax2 = fig2.add_subplot(1,1,1)

            kls = [] 
            for k in range(len(transformer_prediction)):
                temp_b = bayesian[k]+0.001
                temp_t = transformer_prediction[k]+0.001
                res = scipy.special.kl_div(temp_t, temp_b)
                #print (temp_b, temp_t, res)
                kls.append(res)

            #
            x_plot = np.arange(0,11,1)
            ax2.plot(x_plot, kls, c='darkblue', label='kl: bayesian vs. transformer')

            ax2.set_ylim(-.25,.25)
            ax2.plot([0,11],[0,0],'--',c='grey')
            if ctr==0:
                ax2.legend()
                ax2.set_ylabel("KL divergence (2 scalars)")
                ax2.set_xlabel(" coin toss")

          #
          #if self.show_frequentist:
            kls = [] 
            for k in range(1,len(transformer_prediction),1):
                temp_t = transformer_prediction[k]+0.001
                temp_f = frequentist[k-1]+0.001
                res = scipy.special.kl_div(temp_t, temp_f)
                kls.append(res)

            #
            x_plot = np.arange(1,11,1)
            ax2.plot(x_plot, kls, c='darkred', label='kl: frequentist vs. transformer')

          
         # print (self.bt.a0,self.bt.b0)

        plt.suptitle(str(self.bt.n_training_epochs) + " training batch; "+
					 str(self.bt.n_episodes_batch) + " episodes per batch; " +str(self.bt.n_layers) + "-Layer transformer")
        
        if self.save_svg:
            plt.savefig('/home/cat/'+str(self.bt.n_training_epochs)+'_training_epochs_'+
                        str(self.bt.n_episodes_batch)+'_episodes_per_batch_'+
                        str(self.bt.n_layers)+'_layers_'
                        +str(self.bt.params)+ ".svg", format='svg')
        plt.show()

    #
    def cleanup_examples(self):
    
        # remove inf from the KL metrics
        yb = np.array(self.kls_bayesian)
        idx = ~np.isfinite(yb)
        yb[idx] = np.nan

        #
        yb1 = np.nanmean(yb,axis=1)

        yf = np.array(self.kls_frequentist)
        idx = ~np.isfinite(yf)
        yf[idx] = np.nan
        
        #
        self.yb = yb
        self.yf = yf
    
    def get_accuracy(self,k, plotting=False):


        # trans accruacy as the average distance between the bias and the prediction
        accuracy_trans = self.biases.squeeze()[:,None] - self.trans_pred
        self.accuracy_trans = np.mean(np.abs(accuracy_trans), axis=1)
        #self.accuracy_trans = np.mean(accuracy_trans)

        #accuracy_bayes = self.biases.squeeze()[:,None] - self.bayes_pred
        #accuracy_bayes = np.mean(np.abs(accuracy_bayes), axis=1)
        #self.accuracy_bayes = np.mean(accuracy_bayes)

        # same for frequentist
        #accuracy_frequentist = self.biases.squeeze()[:,None] - self.freq_pred
        #accuracy_frequentist = np.mean(np.abs(accuracy_frequentist), axis=1)
        #self.accuracy_freq = np.mean(accuracy_frequentist)

        
    #
    def get_kl_divergences(self):
        
        
        # compute KL divergence between bayesian and transformer predictions
        self.kls_bayesian = scipy.special.kl_div(self.trans_pred, self.bayes_pred)

        # same for frequnetist
        self.kls_freq = scipy.special.kl_div(self.trans_pred[:,1:], self.freq_pred)

    #
    def get_kl_divergence(self, plotting=False):

        #
        fontsize=20

        #
        yb_mean = np.nanmean(self.yb,axis=0)
        self.mean_kl_btrans = yb_mean
        yb_std = np.nanstd(self.yb,axis=0)

        if plotting:
            plt.figure(figsize=(5,5))
            ax1=plt.subplot(111)
            plt.plot(yb_mean, c= 'red', label='mean(KL(trans,bayesian))')
            ax1.fill_between(np.arange(11), yb_mean - yb_std, yb_mean + yb_std,
                            color='red', alpha=.1)
        #
        yf_mean = np.nanmean(self.yf, axis=0)
        self.mean_kl_ftrans = yf_mean
        yf_std = np.nanstd(self.yf,axis=0)
        x_plot = np.arange(1,11,1)

        if plotting:
            plt.plot(x_plot, yf_mean, c= 'blue',  label='mean(kl(freq,trans))')
            ax1.fill_between(x_plot, yf_mean - yf_std, yf_mean + yf_std,
                            color='blue', alpha=.1)
            plt.plot([0,self.n_trials],[0,0],'--',c='grey')
            plt.xlabel("coin toss #", fontsize=fontsize)
            plt.ylabel("mean kl at each time step", fontsize=fontsize)
            plt.legend(fontsize=10)
            plt.title("10 layer transformer")
            ax1.tick_params(axis='both', which='both', labelsize=fontsize)
            plt.show()

    #
    def get_L1(self, plotting=False):
                
        #

        # trans
        biases = np.array(self.biases).squeeze()
        temp = np.array(self.trans_pred)
        temp = np.abs(temp - biases[:,None])
        tt_mean = np.mean(temp, axis=0)
        tt_std = np.std(temp,axis=0)
        self.L1_trans = tt_mean
        if plotting:
            plt.figure(figsize=(15,5))
            ax1=plt.subplot(141)
            plt.plot(tt_mean, c= 'red', label='trans L1')

        #
        temp = np.array(self.bayes_pred)
        temp = np.abs(temp - biases[:,None])
        tt_mean = np.mean(temp, axis=0)
        tt_std = np.std(temp,axis=0)
        self.L1_bayes = tt_mean
        if plotting:
            plt.plot(tt_mean, c= 'blue', label='bayes L1')

        #
        temp = np.array(self.freq_pred)
        temp = np.abs(temp - biases[:,None])

        tt_mean = np.mean(temp, axis=0)
        tt_std = np.std(temp,axis=0)
        self.L1_freq = tt_mean
        x_plot = np.arange(1,11,1)
        if plotting:
            plt.plot(x_plot,tt_mean, c= 'green', label='freq L1')

            plt.xlabel("coin toss #")
            plt.ylabel("L1 (= abs(prediction - coin_bias)) ")
            plt.legend()


    def plot_comparisons(self):

        #
        a0 = self.params[0]
        b0 = self.params[1]
        #n_layers = 100
        #n_training_epochs = 1000
        #n_episodes_batch = 100

        #
        n_layers_array = self.n_layers_array
        n_training_epochs_array = self.n_training_epochs_array

        clrs=['black','blue','red','green']
        cmap = plt.get_cmap('viridis', 4)

        #
        fig1 = plt.figure(figsize=(6,6))
        ax1=plt.subplot(111)

        #
        ctr=0

        #
        for n_training_epochs in n_training_epochs_array:
            xs = []          # number of layers
            yb = []          # bayesian
            yf = []          # frequentist

            #
            for n_layers in n_layers_array:
                #
                bt = BayesianTransformer(n_layers,
                                         n_training_epochs,
                                         self.n_episodes_batch,
                                         self.n_trials,
                                         self.game_type,
                                         self.params,
                                         self.dice_roll_params,
                                         self.len_episode,
                                         random_seed=True) 

                # load model names (also accounts for older model names)
                bt.make_model_name()

                #
                bt.load_model()
                fname_in = bt.fname_model.replace('.btrans','.npz')

                # 
                bt = None
                vis = VisualizeBT_coin_toss(bt)
                vis.load_run(fname_in)

                #
                vis.cleanup_examples()
                
                #
                vis.get_kl_divergence()
                
                #
                #vis.get_L1()

                # load x value
                xs.append(n_layers)
                
                # load mean kl 
                yb.append(np.mean(np.abs(vis.mean_kl_btrans)))
                yf.append(np.mean(np.abs(vis.mean_kl_ftrans)))
            
                               # make the axis labels larger
            ax1.tick_params(axis='both', which='both', labelsize=18)

            #
            ax1.scatter(xs,yb, c=cmap(ctr))
            ax1.scatter(xs,yf, c=cmap(ctr))
            ax1.plot(xs,yb,  
                    linewidth=3,
                    label =" bayes vs trans, # train epoch: "+str(n_training_epochs), c=cmap(ctr))
            ax1.plot(xs,yf,'--', 
                    linewidth=3,
                    label =" freq vs trans, # train epoch: "+str(n_training_epochs),c=cmap(ctr))
            ax1.set_title("Mean abs(KL div) between model predictions")
            ax1.set_xlabel("# of layers")
            ax1.set_ylabel("mean(abs(kl(a,b)))")
            if self.show_legend:
                ax1.legend(fontsize=6)

            ax1.semilogy()
            ax1.semilogx()
            xticks = [1,10,100]
            ax1.set_xticks(xticks)
            ax1.set_title("Beta latent alpha, beta: "+str(self.params))

            ctr+=1

        
        fig1.savefig('/home/cat/kl_'+str(self.params)+'.svg')
        
        plt.show()

    #
    def plot_bayesianity(self):

        # plot the ratio between the kl divergence and the L1 distance
        plt.figure(figsize=(6,6))
        ax=plt.subplot(1,1,1)
        
        # increase font size for all ticks
        ax.tick_params(axis='both', which='both', labelsize=18)


        # make cmap for plotting 4 colors
        cmap = plt.cm.get_cmap('Accent', 4)

        #
        #n_layers_array = [1,10,100]
        #n_training_epochs = 10000
        for ctr, params in enumerate(self.params_list):
            #n_layers = 100

            ratios = []
            #for n_layers in self.n_layers_array:
            for n_training_epochs in self.n_training_epochs_array:
                        
                #
                bt = BayesianTransformer(self.n_layers,
                                         n_training_epochs,
                                         self.n_episodes_batch,
                                         self.n_trials,
                                         self.game_type,
                                         params,
                                         self.dice_roll_params,
                                         self.len_episode,
                                         random_seed=True) 

                # load model names (also accounts for older model names)
                bt.make_model_name()

                #
                bt.load_model()
                fname_in = bt.fname_model.replace('.btrans','.npz')

                # 
                vis = VisualizeBT_coin_toss()
                vis.load_run(fname_in)
                vis.cleanup_examples()
                vis.get_kl_divergence()
                vis.get_L1()

                # load x value
                
                # load mean kl 
                yb = np.mean(np.abs(vis.mean_kl_btrans))
                yf = np.mean(np.abs(vis.mean_kl_ftrans))
                               
                # get ratio of yb and yf
                ratio = np.array(yf)/np.array(yb)
                ratios.append(ratio)

            
            plt.scatter(np.array(self.n_training_epochs_array), 
                        ratios,
                        color=cmap(ctr)
                        )
            
            
            plt.plot(np.array(self.n_training_epochs_array),
                     ratios,
                        linewidth=3,
                        c=cmap(ctr),
                        label = r'$\beta$('+str(self.params_list[ctr][0])+
                        ","+str(self.params_list[ctr][1])+')')
            
        #
        plt.legend(fontsize=12)
        # use the same xticks as the previous plot
        # relabel x axis with params
        #labels = []
        #for k in range(len(self.params_list)):
        #    labels.append(r'$\beta$('+str(self.params_list[k][0])+","+str(self.params_list[k][1])+')')
        #plt.ylabel("Ratio")
        #plt.xticks(np.arange(len(self.params_list)), labels, rotation=15)
        plt.xlabel("# training epochs")

        plt.ylim(bottom=0.5)
        plt.semilogy()
        plt.semilogx()
        # print title with n training epochs
        # plt.title("dist type: "+self.plot_type+",# training epochs: "
        #           +str(self.n_training_epochs_array))
        plt.title("# layers: "+str(self.n_layers))
        plt.show()
        plt.savefig('/home/cat/bayesianity_'+str(self.plot_type)+'.svg')



    #
    def plot_ratios(self):

        if self.plot_type == 'kl':
            self.text = 'kl'
        else:
            self.text = 'L1'


        clrs = ['red','blue','black','green']
        er=[]
        dr=[]
        lr=[]
        for ctr, params in enumerate(self.params_list):
            
            # load the params from the file
            data = np.load('/home/cat/'+str(self.text)+'_ratios_'+str(params)+'.npz')
            er.append(data['er'])
            dr.append(data['dr'])
            lr.append(data['lr'])

        # plot the results
        plt.figure(figsize=(6,6))
        # increase font of ticks
        plt.tick_params(axis='both', which='both', labelsize=18)
        plt.plot(er, c=clrs[0],label="emergence",
                 linewidth=3)
        plt.plot(dr, c=clrs[1],label="learning",
                 linewidth=3)
        plt.plot(lr, c=clrs[2],label="depth",
                 linewidth=3)
        plt.legend()

        # relabel x axis with params
        labels = []
        for k in range(len(self.params_list)):
            labels.append(r'$\beta$('+str(self.params_list[k][0])+","+str(self.params_list[k][1])+')')
        plt.ylabel("Ratio")
        plt.xticks(np.arange(len(self.params_list)), labels, rotation=15)
        plt.ylim(bottom=0)
        plt.plot([0,3],[1,1],'--',c='grey')
        plt.savefig('/home/cat/ratios_'+str(self.text)+'.svg')
        plt.show()




############################################
############################################
############################################
def train_models_wrapper(n_layers_array,
                         n_training_epochs_array,
                         n_episodes_batch,
                         n_trials,
                         predict_during_training_interval,
                         game_type,
                         params_list,
                         n_test_trials,
                         plotting = False,
                         predict_during_training = False,
                         use_multi_distributions = False):

    #self.n_test_trials = n_test_trials

    #
    for n_layers in n_layers_array:
        for n_training_epochs in n_training_epochs_array:
            for params in params_list:

                #
                coin_toss_params = dice_roll_params = params
                #dice_roll_params = dice_roll_params_list
                
                #
                bt = BayesianTransformer(n_layers,
                                         n_training_epochs,
                                         n_episodes_batch,
                                         n_trials,
                                         game_type,
                                         coin_toss_params,
                                         dice_roll_params)        
                #
                bt.n_test_trials = n_test_trials

                # if using multi distributions
                bt.params_list = params_list
                bt.use_multi_distributions = use_multi_distributions

                #
                bt.predict_during_training = predict_during_training
                bt.predict_during_training_interval = predict_during_training_interval

                #
                if game_type == "coin_toss":
                    if bt.use_multi_distributions:
                        bt.fname_model = os.path.join(bt.root_dir,
                                            'model_nlayers_'+str(n_layers)
                                            +"_ntraining_"+str(n_training_epochs)
                                            +"_ncoin_tosses_"+str(n_trials)
                                            +"_coin_toss_multi_"
                                            +str(params_list)+".btrans")
                    else:
                        bt.fname_model = os.path.join(bt.root_dir,
                                            'model_nlayers_'+str(n_layers)
                                            +"_ntraining_"+str(n_training_epochs)
                                            +"_ncoin_tosses_"+str(n_trials)
                                            +"_coin_toss_"
                                            +str(params)+".btrans")
                elif game_type == "dice_roll":
                    bt.fname_model = os.path.join(bt.root_dir,
                                            'model_nlayers_'+str(n_layers)
                                            +"_ntraining_"+str(n_training_epochs)
                                            +"_ndice_rolls_"+str(n_trials)
                                            +"_dice_roll_"
                                            +str(dice_roll_params)+".btrans")
                
                # MAKE AND TRAIN MODEL
                if os.path.exists(bt.fname_model)==False:

                    #
                    bt.init_model()

                    #
                    bt.plotting = plotting

                    #
                    if game_type == "coin_toss":
                        bt.run_model_coin_toss()
                    #
                    elif game_type == "dice_roll":
                        bt.run_model_dice_roll()
                    #
                    else:
                        print ("Model not found: ")
                        return

                    #
                    bt.save_model()
                else:
                    print ("Model exists: ", bt.fname_model)
                    bt.load_model()

                # PREDICT MODEL
                if game_type == "coin_toss":
                    vis = VisualizeBT_coin_toss(bt)
                    vis.generate_examples_coin_toss(bt.n_test_trials)
                    vis.save_run_coin_toss(bt.fname_model[:-7]+".npz")

                #
                elif game_type == "dice_roll":
                    vis = Visualize_dice_roll(bt)
                    vis.generate_examples_dice_roll(bt.n_test_trials)
                    vis.save_run_dice_roll(bt.fname_model[:-7]+".npz")
            
                #
