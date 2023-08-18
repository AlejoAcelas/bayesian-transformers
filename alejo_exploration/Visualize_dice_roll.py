
#
#from transformer_lens import EasyTransformer, EasyTransformerConfig
import tqdm.auto as tqdm
from scipy.stats import dirichlet

#
from tqdm import trange

#
import torch
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"
import scipy, os
import numpy as np

#
class Visualize_dice_roll():
   
    #
    def __init__(self, bt=None):

        #
        self.bt = bt

    #
    def save_run_dice_roll(self, fname_out):

        try: 
           print ("losses: ", self.bt.losses) 
        except:
           self.bt.losses = None

        #
        np.savez(fname_out,
                 kls_bayesian = self.kls_bayesian,
                 #kls_frequentist = self.kls_frequentist,
                 trans_pred = self.trans_pred,
                 bayes_pred = self.bayes_pred,
                 #freq_pred = self.freq_pred,
                 biases = self.biases,
                 losses = self.bt.losses
                )
#
    def generate_examples_dice_roll(self, n_trials):
        import scipy
        
        #
        self.biases = []
        kls_bayesian = []
        kls_frequentist = []
        trans_pred = []
        bayes_pred = []
        freq_pred = []

        # TODO: simplify this to do all batches in 1 step...
        for ctr,test_id in enumerate(trange(n_trials, desc='testing')):

              test_data, bias = self.bt.make_data_generator_dice_roll(1)
              #print ("test_data: ", test_data)
              self.biases.append(bias)

              #
              #######################################################
              ############### TRANSFORMER RESULTS ###################
              #######################################################
              test_data = test_data[0]
              test_data_nobos = test_data[1:]
              logits = self.bt.model(test_data)

              #
              logits = logits.cpu().detach().numpy()[0]
              transformer_prediction = []
              for logit in logits:
                temp = np.exp(logit)/sum(np.exp(logit))
                temp = temp[0]/np.sum(temp)
                transformer_prediction.append(temp)

              trans_pred.append(transformer_prediction)

              #######################################################
              ################# PLOT BAYESIAN OPTIMAL RESULTS #######
              #######################################################
              # update prior using bernoulli update rule inference
              bayesian = []
              confs = []
                
              # generate first value
              #dist = scipy.stats.beta(self.bt.a0,self.bt.b0)
              # for dice roll start with uniform prior
              alpha_post = np.ones(3)

              dist = dirichlet.mean(alpha_post)
              bayesian.append(dist.mean())

              #interval = dist.interval(0.95)
              #confs.append(interval)

              #
              #print ("test_data_nobos: ", test_data_nobos.shape)
              for q in range(0,test_data_nobos.shape[0],1):
                outcome = test_data_nobos[q]#.numpy()
                #print ("outcome of dice roll: ", outcome)
                
                #
                #alpha_post = alpha_post + np.bincount(outcome, minlength=3)
                alpha_post[outcome]+=1
                theta_post = dirichlet.mean(alpha_post)#, size=1).squeeze()

                #
                #dist = scipy.stats.beta(a,b)
                theta_post = dirichlet.mean(alpha_post)
                bayesian.append(theta_post)

                #interval = dist.interval(0.95)
                #confs.append(interval)

              #
              bayesian = np.array(bayesian)
              bayes_pred.append(bayesian)

              #print ("bayesian: ", bayesian)

            #   #######################################################
            #   ################# PLOT FREQUENTIST RESULTS ############
            #   #######################################################
            #   # update prior using bernoulli update rule inference
            #   x_plot = np.arange(1,11,1)

            #   #
            #   frequentist = []
            #   for q in range(0,test_data_nobos.shape[0],1):
            #     outcome = test_data_nobos[:q+1]

            #     idx = np.where(outcome.cpu().numpy()==1)[0]
            #     frequentist.append(1-idx.shape[0]/len(outcome))

            #   freq_pred.append(frequentist)

              ###############################################################
              ####################### KL ENTROPY ############################
              ###############################################################

              kls = [] 
              for k in range(len(transformer_prediction)):
                temp_b = bayesian[k] #+0.00001
                temp_t = transformer_prediction[k] #+0.00001
                res = scipy.special.rel_entr(temp_t, temp_b)
                #print (temp_b, temp_t, res)
                kls.append(res)

              kls_bayesian.append(kls)

            #   #
            #   kls = [] 
            #   for k in range(1,len(transformer_prediction),1):
            #     temp_t = transformer_prediction[k]#+0.00001
            #     temp_f = frequentist[k-1]#+0.00001
            #     res = scipy.special.rel_entr(temp_t, temp_f)
            #     kls.append(res)
            #   kls_frequentist.append(kls)

        #
        self.kls_bayesian = kls_bayesian
        #self.kls_frequentist = kls_frequentist
        self.trans_pred = trans_pred
        self.bayes_pred = bayes_pred
        #self.freq_pred = freq_pred
        #self.biases = biases

    def load_run(self, fname_in):

        d = np.load(fname_in)

        self.kls_bayesian = d['kls_bayesian']
        self.kls_frequentist = d['kls_frequentist']
        self.trans_pred = d['trans_pred']
        self.bayes_pred = d['bayes_pred']
        self.freq_pred = d['freq_pred']
        self.biases = d['biases']

    #
    def show_examples(self, n_examples):
		
        #
        import scipy

        clrs=['green','orange','blue']
        
        #
        test_ids = np.random.choice(np.arange(100), n_examples, replace=False)

        #
        fig1 = plt.figure(figsize=(6,6), facecolor='white')
        plt.suptitle("Beta("+str(self.bt.a0)+","+str(self.bt.b0)+")")
        
        #
        if False:
          fig2 = plt.figure(figsize=(6,6), facecolor='white')
        
        #
        for ctr,test_id in enumerate(test_ids):

          #
          test_data, bias = self.bt.make_data_generator_dice_roll(1)
          print ("test_data: ", test_data, ", bias: ", bias)
          #######################################################
          ############### TRANSFORMER RESULTS ###################
          #######################################################
          test_data = test_data[0]
          test_data_cpu = test_data.cpu() #.detach().numpy().to("cpu")
          test_data_nobos = test_data[1:].cpu().detach().numpy()
          logits = self.bt.model(test_data_cpu).squeeze()

          #
          ##!! Please replace the below with torch.nn.functional.softmax(logits, dim=1)
           # OK DONE replaced with softmax below
          logits = torch.nn.functional.softmax(logits, dim=1)[:,:3]

          #
          transformer_prediction = logits.cpu().detach().numpy()

          ##################################
          ##################################
          ##################################
          x_plot = np.arange(0,11,1)
          ax1 = fig1.add_subplot(1,1,1)
          for k in range(3):
            ax1.plot(x_plot, transformer_prediction[:,k], c=clrs[k], alpha=0.5,
                     linewidth=3)
            ax1.scatter(x_plot, transformer_prediction[:,k], c= clrs[k], s=10)

          #######################################################
          ################# PLOT BAYESIAN OPTIMAL RESULTS #######
          #######################################################
          # update prior using bernoulli update rule inference

          #if False:
          bayesian = []

          # 
          # start with uniform distribution or ground truth distribuiont? 
          starting_alpha = np.ones(3)
          starting_alpha = np.array(self.bt.params)
          alpha_post = starting_alpha.copy()
          traces = []
          traces.append(alpha_post.copy())
          for k in range(test_data_nobos.shape[0]):
             
            obs = test_data_nobos[k]#.numpy()
            #print ("obs: ", obs)

            # 
            alpha_post[obs] += 1
            #theta_post = dirichlet.rvs(alpha_post, size=1).squeeze()
            theta_post = dirichlet.mean(alpha_post)#, size=1).squeeze()

            # 
            traces.append(theta_post)

          traces = np.vstack(traces)
          #print ("traces: ", traces.shape)
          for k in range(3):
            plt.plot(x_plot[:], traces[:,k], '--',
                   c=clrs[k], linewidth=2)

          # #
          # bayesian = np.array(bayesian)
          # ax1.plot(x_plot,bayesian, c='green',label='bayesian (shading 95% conf int)')
          # ax1.scatter(x_plot,bayesian, c='green')

          # confs=np.vstack(confs)

          # # #
          # ax1.fill_between(x_plot, bayesian - confs[:,0], bayesian + confs[:,-1],
          #               color='green', alpha=.1)

          #####################################################
          ################ DICE ROLL PLOTS ####################
          #####################################################
          # heads
          x = np.where(test_data_cpu==0)[0]-0.5
          offset = -0.2
          sett = 0.05
          ax1.vlines(x, ymin=x*0+offset, ymax=x*0 +offset-sett, colors=clrs[0], lw=2)
          offset+=0.1
          # tails
          x = np.where(test_data_cpu==1)[0] - 0.5
          ax1.vlines(x, ymin=x*0+offset, ymax=x*0+offset-sett, colors=clrs[1], lw=2)
          offset+=0.1

          # tails
          x = np.where(test_data_cpu==2)[0] - 0.5
          ax1.vlines(x, ymin=x*0+offset, ymax=x*0+offset-sett, colors=clrs[2], lw=2)
          
          # plot bias
          for ctr3, b_ in enumerate(bias[0]):
            print ("b_: ", b_)
            ax1.fill_between(x_plot, b_ - 0.02, b_ + 0.02,
                              color=clrs[ctr3], alpha=.1)



          #######################################################
          ################# PLOT FREQUENTIST RESULTS ############
          #######################################################
          if True:
            # update prior using bernoulli update rule inference
            x_plot = np.arange(1,11,1)

            # start counting after the first dice roll
            frequentist = np.zeros(3)
            probs = np.zeros((10,3))
            for q in range(0,test_data_nobos.shape[0],1):
              outcome = test_data_nobos[q]

              frequentist[outcome] += 1
              probs[q] = frequentist/np.sum(frequentist)
              #

              print (probs[q])  
            #   idx = np.where(outcome==1)[0]
            #   frequentist.append(1-idx.shape[0]/len(outcome))

            # plot frequentist same as bayesian but using different line style
            for q in range(probs.shape[1]):
              ax1.plot(x_plot, probs[:,q], c=clrs[q], alpha=0.5,
                       linewidth=3, linestyle=':')
              ax1.scatter(x_plot, probs[:,q], c= clrs[q], s=10)
               

            # #########################################################
            # if ctr==0:
            #   ax1.legend(loc=1,fontsize=8)
            #   ax1.set_xlabel("prediction # (integers) / toss # (half-integers)")
            #   ax1.set_ylabel("Prob of heads")

            # ax1.set_ylim(-0.2,1.1)
            # ax1.set_title("Coin bias: "+str(round(bias[0][0],4)))

          ###############################################################
          ####################### KL ENTROPY ############################
          ###############################################################
          if False:
            import scipy
            ax2 = fig2.add_subplot(1,6,ctr+1)

            kls = [] 
            for k in range(len(transformer_prediction)):
              temp_b = bayesian[k]+0.001
              temp_t = transformer_prediction[k]+0.001
              res = scipy.special.rel_entr(temp_t, temp_b)
              #print (temp_b, temp_t, res)
              kls.append(res)

            #
            x_plot = np.arange(0,11,1)
            ax2.plot(x_plot, kls, c='darkblue', label='kl: bayesian vs. transformer')

            #
            kls = [] 
            for k in range(1,len(transformer_prediction),1):
              temp_t = transformer_prediction[k]+0.001
              temp_f = frequentist[k-1]+0.001
              res = scipy.special.rel_entr(temp_t, temp_f)
              kls.append(res)

            #
            x_plot = np.arange(1,11,1)
            ax2.plot(x_plot, kls, c='darkred', label='kl: frequentist vs. transformer')

            ax2.set_ylim(-.25,.25)
            ax2.plot([0,11],[0,0],'--',c='grey')
            if ctr==0:
              ax2.legend()
              ax2.set_ylabel("KL divergence (2 scalars)")
              ax2.set_xlabel(" coin toss")

         # print (self.bt.a0,self.bt.b0)
        # increase tick size for all axis
        ax1.tick_params(axis='both', which='both', labelsize=18)


        # 
        plt.ylim(-0.3,1.)
        plt.suptitle(str(self.bt.n_training_epochs) + " training batch; "+
					 str(self.bt.n_episodes_batch) + " episodes per batch; " +str(self.bt.n_layers) + "-Layer transformer")
        plt.savefig('/home/cat/dice_roll_'+str(self.bt.params)+'.svg')
        plt.show()

    
    
    #
    def cleanup_examples(self):
    
        # remove inf from the KL metrics
        yb = np.array(self.kls_bayesian)
        idx = ~np.isfinite(yb)
      #  print ("removing kl(beyasian,tranfsomer) vals #: ", idx.sum())
        yb[idx] = np.nan

        #
        yb1 = np.nanmean(yb,axis=1)

        yf = np.array(self.kls_frequentist)
        idx = ~np.isfinite(yf)
       # print ("removing kl(ferquentist,tranfsomer) vals #: ", idx.sum())
        yf[idx] = np.nan
        
        #
        self.yb = yb
        self.yf = yf
    
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
            plt.plot([0,10],[0,0],'--',c='grey')
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

        #############################
        # ax1=plt.subplot(142)
        #
        # biases = np.array(self.biases).squeeze()
        # #
        # temp = np.array(self.trans_pred)
        # temp = np.abs(temp - biases[:,None])
        #
        # tt_mean = np.mean(temp, axis=0)
        # tt_std = np.std(temp,axis=0)
        # plt.plot(tt_mean, c= 'red', label='trans L1')
        # ax1.fill_between(np.arange(11), tt_mean - tt_std, tt_mean + tt_std,
        #                 color='red', alpha=.1)
        # plt.plot([0,10],[0,0],'--',c='grey')
        # plt.ylim(-0.1,1)
        # plt.legend()
        #
        # ###########################
        # ax1=plt.subplot(143)
        # temp = np.array(self.bayes_pred)
        # temp = np.abs(temp - biases[:,None])
        #
        # tt_mean = np.mean(temp, axis=0)
        # tt_std = np.std(temp,axis=0)
        # plt.plot(tt_mean, c= 'blue', label='bayes L1')
        # ax1.fill_between(np.arange(11), tt_mean - tt_std, tt_mean + tt_std,
        #                 color='blue', alpha=.1)
        # plt.plot([0,10],[0,0],'--',c='grey')
        # plt.ylim(-0.1,1)
        # plt.legend()
        #
        # #############################
        # ax1=plt.subplot(144)
        # temp = np.array(self.freq_pred)
        # temp = np.abs(temp - biases[:,None])
        #
        # tt_mean = np.mean(temp, axis=0)
        # tt_std = np.std(temp,axis=0)
        # x_plot = np.arange(1,11,1)
        # plt.plot(x_plot,tt_mean, c= 'green', label='freq L1')
        # ax1.fill_between(x_plot, tt_mean - tt_std, tt_mean + tt_std,
        #                 color='green', alpha=.1)
        #
        # plt.ylim(-0.1,1)
        # plt.plot([0,10],[0,0],'--',c='grey')
        #
        # plt.legend()


    def plot_comparisons(self):

        #
        a0 = self.params[0]
        b0 = self.params[1]
        #n_layers = 100
        #n_training_epochs = 1000
        #n_episodes_batch = 100

        n_layers_array = [1,10,100]
        n_training_epochs_array = [1,100,1000,10000]

        clrs=['black','blue','red','green']
        cmap = plt.get_cmap('viridis', 4)

        #
        if self.plotting:
          if self.show_kl:
            #fig1 = plt.figure(figsize=(14,5))
              fig1 = plt.figure(figsize=(6,6))
              ax1=plt.subplot(111)
              #ax2=plt.subplot(122)
            #fig1, ((ax1, ax2)) = plt.subplots(1, 2)

          if self.show_L1:
              fig2 = plt.figure(figsize=(6,6))
              ax3=plt.subplot(111)
              #ax4=plt.subplot(122)

        # track the results from the single layer 10k training vs. 100 layer 1k training
        l1_e10000 = None
        l100_e1000 = None
        l100_e10000 = None

        #
        ctr=0

        #
        for n_training_epochs in n_training_epochs_array:
            xs = []
            yb = []
            yf = []
            l1_tb = []
            l1_tf = []
            #l1_f = []
            for n_layers in n_layers_array:
                #
                fname_in = os.path.join('/media/cat/4TBSSD/btrans',
                                        'model_nlayers_'+str(n_layers)
                                        +'_ntraining_'+str(n_training_epochs)
                                        +'_a0_'+str(a0)
                                        +'_b0_'+str(b0)
                                        +'.npz')

                # 
                bt = None
                vis = VisualizeBT_coin_toss(bt)
                vis.load_run(fname_in)
                vis.cleanup_examples()
                vis.get_kl_divergence()
                vis.get_L1()

                # load x value
                xs.append(n_layers)
                
                # load mean kl 
                yb.append(np.mean(np.abs(vis.mean_kl_btrans)))
                yf.append(np.mean(np.abs(vis.mean_kl_ftrans)))
                
                # load mean L1
                l1_tb.append(np.mean(np.abs(vis.L1_trans-vis.L1_bayes)))
                l1_tf.append(np.mean(np.abs(vis.L1_trans[1:] - vis.L1_freq)))
            
                # find the 10k training epoch for 1 layer
                if n_training_epochs==10000 and n_layers==1:
                    if self.show_kl:
                       l1_e10000 = np.mean(np.abs(vis.mean_kl_btrans))
                    else:
                       l1_e10000 = np.mean(np.abs(vis.L1_trans-vis.L1_bayes))

                # find the 10k training epoch for 1 layer
                if n_training_epochs==10000 and n_layers==100:
                    if self.show_kl:
                       l100_e10000 = np.mean(np.abs(vis.mean_kl_btrans))
                    else:
                       l100_e10000 = np.mean(np.abs(vis.L1_trans-vis.L1_bayes))

                # also find the 1k training epoch for 100 layers:
                if n_training_epochs==1000 and n_layers==100:
                    if self.show_kl:
                       l100_e1000 = np.mean(np.abs(vis.mean_kl_btrans))
                    else:
                       l100_e1000 = np.mean(np.abs(vis.L1_trans-vis.L1_bayes))
                       
                
            #
            #print (n_training_epochs, n_layers)
            
            # plot multi metrics
            if self.show_kl and self.plotting:
              #ax1 = plt.subplot(1,2,1)
              ax1.scatter(xs,yb, c=cmap(ctr))
              ax1.scatter(xs,yf, c=cmap(ctr))
              ax1.plot(xs,yb,  label =" bayes vs trans, # train epoch: "+str(n_training_epochs), c=cmap(ctr))
              ax1.plot(xs,yf,'--', label =" freq vs trans, # train epoch: "+str(n_training_epochs),c=cmap(ctr))
              ax1.set_title("Mean abs(KL div) between model predictions")
              ax1.set_xlabel("# of layers")
              ax1.set_ylabel("mean(abs(kl(a,b)))")
              if self.show_legend:
                  ax1.legend(fontsize=6)
              #ax1.set_ylim(bottom=0)
              #plt.legend(fontsize=6)
              ax1.semilogy()
              ax1.semilogx()
              xticks = [1,10,100]
              ax1.set_xticks(xticks)
              ax1.set_title("Beta latent alpha, beta: "+str(self.params))

            # plot L1 metrics
            if self.show_L1 and self.plotting:
              #ax3 = plt.subplot(1,2,2)
              ax3.scatter(xs, l1_tb,  c=cmap(ctr))
              ax3.plot(xs,l1_tb, label =" bayes vs trans, # train epoch: "+str(n_training_epochs), c=cmap(ctr))
              ax3.scatter(xs, l1_tf, c=cmap(ctr))
              ax3.plot(xs,l1_tf,'--', label =" freq vs. trans, # train epoch: "+str(n_training_epochs),  c=cmap(ctr))
              plt.title("Mean L1 distance between model predictions")
              plt.xlabel("# of layers")
              plt.ylabel("ave(abs(L1(a,b)))")

              xticks = [1,10,100]
              ax3.set_xticks(xticks)
              #plt.legend(fontsize=6)
              ax3.set_title("Beta latent alpha, beta: "+str(self.params))
              plt.semilogy()
              plt.semilogx()
            
            #
            ctr+=1

        # compute ratios
        er = l100_e1000/l1_e10000
        dr = l100_e1000/l100_e10000
        lr = l1_e10000/l100_e10000
        print ("Emergence ratio :", er)
        print ("Depth ratio: ", dr)
        print ("Learning ratio: ", lr)
        print (" ")
        # save these ratios
        if self.show_kl:
          np.savez('/home/cat/kl_ratios_'+str(self.params)+'.npz',
                    er=er,
                    dr=dr,
                    lr=lr)

        if self.show_L1:
          np.savez('/home/cat/L1_ratios_'+str(self.params)+'.npz',
                    er=er,
                    dr=dr,
                    lr=lr)

        #
        if self.plotting:
          
          #plt.ylim(bottom=0)
          #plt.plot([0,100],[0,0],'--',c='grey')
          #plt.xlim(left=0)
          if self.show_kl:
              fig1.savefig('/home/cat/kl_'+str(self.params)+'.svg')
          if self.show_L1:
              fig2.savefig('/home/cat/L1_'+str(self.params)+'.svg')
          plt.show()


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
        plt.figure()
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

