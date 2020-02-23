# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 19:27:22 2020

@author: Parviz.Asoodehfard
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed
from copy import deepcopy
import math
from time import time
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import warnings
warnings.filterwarnings("ignore")

#checklist
# review all variable name
# convert all numbers to variable
# 



class _chromosome():
    
    def __init__(self,n_features,prc_features_rand_init):
        self.mutated=False
        self.n_features = n_features
        self.genom = np.random.rand(n_features) < prc_features_rand_init
        self.pure_score = 0
        self.with_penalty_score =0
        self.before_mutation_score = 0
        self.min_parent_score = None
        self.max_parent_score = None
        self.generation_lived_longer=0
        self.diversity_score=math.inf
        self.chromosome_age=0
    
    def create_chromosome_by_genom(self,genom):
        self.genom = genom

    def mutate(self,mutate_rate_micro):
        # a mask to versa vise some gens
        mask = np.random.rand(self.n_features) < mutate_rate_micro 
        self.genom[mask] = ~ self.genom[mask] 
        self.mutated = True
        self.before_mutation_score=self.with_penalty_score
        return self
    
    def mutate_ex_crr(self,corr):
        # create separate index for True and False in genom
        idx_active_gens=[i for i in range(self.n_features) if self.genom[i]]
        idx_deactive_gens=[i for i in range(self.n_features) if not self.genom[i]]
        
        #choice some gens form active part of genom and turn off them
        gens_to_turn_off=np.random.choice(idx_active_gens,1)[0]
        self.genom[gens_to_turn_off] = False         
        
        # turn on some deactive gen, but more correlated to previous choise should have more chanse
        corr2_deactives=np.array(corr.iloc[gens_to_turn_off,idx_deactive_gens]**2)
        probility=corr2_deactives/sum(corr2_deactives)
        gens_to_turn_on = np.random.choice(idx_deactive_gens,1,p=probility)
        self.genom[gens_to_turn_on] = True
        
        
        self.mutated = True
        self.before_mutation_score=self.with_penalty_score
        
        return self
        
    def crossover(self,ch1,ch2,f_periority_in_crossover):
        # if f is turn of or ch2 has better score
        if ch2.with_penalty_score > ch1.with_penalty_score or (not f_periority_in_crossover):
            chromosome1=ch1
            chromosome2=ch2
        else:
            chromosome1=ch2
            chromosome2=ch1
        
        self.genom = deepcopy(chromosome1.genom)
        # 0.7 from chromosome2 and 0.3 from chromosome1
        mask = np.random.rand(self.n_features) > 0.3
        self.genom[mask] = chromosome2.genom[mask]
        
        #set parent scores
        self.min_parent_score = min(chromosome1.with_penalty_score,chromosome2.with_penalty_score)
        self.max_parent_score = max(chromosome1.with_penalty_score,chromosome2.with_penalty_score)
        
        return self
    
    def set_score(self,estimator_score,l1):
        # it helps to have better score with lower feature 
        l1_penalty = sum(self.genom) * l1 
        self.pure_score = estimator_score
        self.with_penalty_score = ( estimator_score * 100 ) - l1_penalty
        
        # if it has no parent like the first generation fill parent score with itself score's
        if self.min_parent_score is None:
            self.max_parent_score=self.with_penalty_score
            self.min_parent_score=self.with_penalty_score
        
    def update_diversity_score(self,c):
        contradictory_with_active_gens = np.logical_and(c.genom , self.invert_genom)
        cnt_contradictory_gens = np.count_nonzero(contradictory_with_active_gens)
        self.diversity_score = min(self.diversity_score,cnt_contradictory_gens)   
        
    def init_diversity_calc(self):
        self.diversity_score=math.inf
        self.invert_genom = np.invert(self.genom)
        





#==============================================================================
# Class performing feature selection with genetic algorithm
#==============================================================================

        
    
class GeneticSelector():

    def __init__(   self, 
                    estimator, scoring='f1',l1=0,cv=10,
                    n_gen=10, n_best=5, n_rand=20, n_children=6, n_die=3,prc_features_rand_init=.05,
                    initial_size_ratio=100,initial_rate_search_score=10,
                    mutation_rate=0.05,mutate_rate_micro=0.001,
                    previous_result=None,
                    n_jobs=0,
                    max_iterate=100,
                    f_diverse=False,
                    f_periority_in_crossover=False,
                    f_extra_best_previous=False,
                    f_extra_bests_previous=False,
                    f_extra_mutate=False,
                    f_extra_crossover_with_minimal=False,
                    f_rfecv=False,
                    f_dynamic_child=False,
                    verbose=False,
                    show_plots=True):

        self.estimator = estimator
        self.scoring = scoring        # method of evaluation
        self.l1 = l1        # l1 regulization
        self.cv=cv
        
        self.n_gen = n_gen        # Number of generations
        self.n_best = n_best       # Number of best chromosomes to select
        self.n_rand = n_rand            # Number of random chromosomes to select
        self.n_children = n_children           # Number of children created during crossover
        self.n_die=n_die        # n worse will die
        self.n_features=0        #set in fit
        self.prc_features_rand_init = prc_features_rand_init
        
        #keep_diversity_of_first_generation
        self.initial_size_ratio = initial_size_ratio # the first generation need to bee diverse, we can just generate a huge population and at first step select in a way to keep diversity which is faster than scoring
        self.initial_rate_search_score = initial_rate_search_score
        
        self.size = (n_best + n_rand) * n_children        # Number of chromosomes in population
        
        self.mutation_rate = mutation_rate        # Probablity of chromosome mutation
        self.mutate_rate_micro = mutate_rate_micro        # Probablity of mutation for each genum
        
        self.previous_result = previous_result        # warm start
        self.n_jobs=n_jobs        # make parallel by thread for each scoring

        self.max_iterate = max_iterate       
        
        
        self.f_diverse=f_diverse
        self.f_periority_in_crossover=f_periority_in_crossover
        self.f_extra_best_previous=f_extra_best_previous
        self.f_extra_bests_previous=f_extra_bests_previous
        self.f_extra_mutate=f_extra_mutate
        self.f_extra_crossover_with_minimal=f_extra_crossover_with_minimal
        self.f_dynamic_child=f_dynamic_child
        self.f_rfecv=f_rfecv
        
        self.verbose_flag = verbose
        self.show_plots = show_plots
        
        self.the_best=_chromosome(self.n_features,self.prc_features_rand_init)

        print('generation size: ',(self.n_best + self.n_rand) * self.n_children)               # number of child in each generation
    
    
    def verbose(self,msg,end=None):
        if self.verbose_flag:
            print(msg,end=end)
            
            
    def initilize(self):
        self.verbose('start initilize')
        population = []

        
        # generate a huge random population (without score)
        for i in range(self.size*self.initial_size_ratio):
            chromosome = _chromosome(self.n_features,self.prc_features_rand_init)
            population.append(chromosome)
            

        population_next=[]
        population_next.append(population.pop(0))        #select the first one and move

        
        #calculate diversity for remained
        for j in range(len(population)):
            population[j].invert_genom = np.invert(population[j].genom)
            population[j].update_diversity_score(population_next[0])
        
        # select 10 times of size form huge population based on diversity (to keep diversity)
        for i in range(self.size*self.initial_rate_search_score): 
            diversity_scores_list = [c.diversity_score for c in population]            #find next rand based on diversity score
            most_different_chromosome = np.argmax(diversity_scores_list)
            population_next.append(population.pop(most_different_chromosome))            #move selected from population to next_population
            #update diversity scores
            for j in range(len(population)):
                population[j].update_diversity_score(population_next[-1])
            if population == []:
                break

                            
        #add the previous result to the population for warm start
        if self.previous_result is not None:
            chromosome = _chromosome(self.n_features,self.prc_features_rand_init)
            chromosome.create_chromosome_by_genom(self.previous_result,self.prc_features_rand_init)#************
            population_next.append(chromosome)

        self.verbose('end initize')
        return population_next
    
    def calc_score(self,chromosome):
        self.verbose('.',end='')
        X, y = self.dataset

        # mean 10 cv score
        estimator_score = np.mean(  cross_val_score(self.estimator, 
                                                    X.loc[:,chromosome.genom], 
                                                    y, 
                                                    cv=self.cv, 
                                                    scoring=self.scoring))
        
        chromosome.set_score(estimator_score,self.l1)#***************

        #score *100 - l1 penalty
        return chromosome.with_penalty_score  #( estimator_score * 100 ) - l1_penalty 

    def fitness(self, population):
        t0 = time()
        # evaluate score of all population with parallel
        scores = Parallel(n_jobs= self.n_jobs,
                          backend="threading")  (  map(delayed(self.calc_score), population)   )   
        
        # reverse the sort (max score will be first)
        scores = np.array(scores)
        population = np.array(population) 
        inds = np.argsort(-scores)
        sorted_scores = list(scores[inds])
        sorted_population = deepcopy(list(population[inds]))
        
        #calc the cost time for fitness
        self.time[-1]['fitness'] = time()-t0
        return sorted_scores,sorted_population
    
    def select(self, pop_sorted):
        self.verbose('start select')
        t0 = time()
        
        #n_best
        population_next = pop_sorted[:self.n_best]
        rest_pop = pop_sorted[self.n_best:-self.n_die]
        
        #n_rand
        
        if self.f_diverse:            # KEEP DIVERSITY
            
            #set diversity scores
            for j in range(len(rest_pop)):
                rest_pop[j].init_diversity_calc()
                for c in population_next:
                    rest_pop[j].update_diversity_score(c)
            
            # create diversity_score order
            self.rest_pop_scores=[]
            self.rest_pop_diversity=[]
            self.rest_pop_mutated=[]
            for rpi in rest_pop:
                self.rest_pop_scores.append(rpi.with_penalty_score)#pure_score)
                self.rest_pop_diversity.append(rpi.diversity_score)
                self.rest_pop_mutated.append(rpi.mutated)
            rp_order = np.argsort(self.rest_pop_scores)+np.argsort(self.rest_pop_diversity)
            
            # for ploting pourpose
            for rpi in pop_sorted[:self.n_best]:
                self.rest_pop_scores.append(rpi.with_penalty_score)#pure_score)
                self.rest_pop_diversity.append(rpi.diversity_score)
                self.rest_pop_mutated.append(rpi.mutated)
                
                
            #SELECT BASED ON DIVERSITY_SCORE ORDER
            for i in range(int(self.n_rand/2)):
                i_max = np.argmax(rp_order)
                rp_order=np.delete(rp_order,i_max)
                population_next.append(rest_pop.pop(i_max))  
            for i in range(int(self.n_rand/2)):
                #find next rand based on xor
                div_scores = [c.diversity_score for c in rest_pop]
                i_max = np.argmax(div_scores)
                #move selected
                population_next.append(rest_pop.pop(i_max))
                #update xor scores
                for j in range(len(rest_pop)):
                    rest_pop[j].update_diversity_score(population_next[-1])


        # select randomly and without keep diversity
        else:
            for i in range(self.n_rand):
                population_next.append( random.choice(pop_sorted[self.n_best:-self.n_die]) )


        self.verbose('end select')
        self.time[-1]['select'] = time()-t0
        return population_next

    def crossover(self, population):
        self.verbose('start crossover')
        t0 = time()
        population_next=[]
        len_pop=len(population)
        
        # greater score will have more child
        if self.f_dynamic_child:
            population_scores_by_order = [p.with_penalty_score for p in population]
            max_chance = self.n_children
            chance_base_order_of_score = [int((i/len_pop)*max_chance) for i in np.argsort(population_scores_by_order)]
            for pi in range(len_pop):
                for ch in range(int(chance_base_order_of_score[pi]+self.n_children)):
                    rand_chromosome=np.random.choice(population)
                    chromosome = _chromosome(self.n_features,self.prc_features_rand_init
                                            ).crossover(rand_chromosome, 
                                                        population[pi],
                                                        self.f_periority_in_crossover)#**********
                    population_next.append(chromosome) 
                    
        # all population will have the same number of child
        else:
            random.shuffle(population)
            for i in range(len_pop):
                # select two chromosome
                chromosome1, chromosome2 = population[i], population[len_pop-1-i]
                for j in range(self.n_children):
                    chromosome = _chromosome(self.n_features,self.prc_features_rand_init
                                            ).crossover(chromosome1, chromosome2,self.f_periority_in_crossover)#**********
                    population_next.append(chromosome)
    
    
        self.verbose('end crossover')    
        self.time[-1]['crossover'] = time()-t0
        return population_next

    def mutate(self, population):
        t0=time()
        
        population_next = []
        for i in range(len(population)):
            if random.random() < self.mutation_rate:
                population[i].mutate(self.mutate_rate_micro)#*************
                self.verbose(',',end='')
            population_next.append(population[i])
            
        self.time[-1]['mutate'] = time()-t0
        return population_next
    
    def extra_append(self,next_population,population_sorted):
        self.verbose('extra_append')
        t0 = time()
        
        
        X, y = self.dataset
        
        # Keep the best
        if population_sorted[0].with_penalty_score>self.the_best.with_penalty_score:
            self.the_best=deepcopy(population_sorted[0])
            
        # if f_extra_best_previous or the best of 5 previous past better than current best, bring the best to new generation    
        try:
            if self.scores_best[-5]>self.scores_sorted[-1] or self.f_extra_best_previous:
                next_population.append(deepcopy(self.the_best))
        except:
            pass

        # give an extra chance to the best
        if population_sorted[0].chromosome_age<1:
            population_sorted[0].chromosome_age+=1
            next_population.append(deepcopy(population_sorted[0]))

        # for all the bests if f_function is true do it
        ips = 0
        while population_sorted[ips].with_penalty_score == population_sorted[0].with_penalty_score:
            ips+=1
            
            # Give a extra change to the bests
            if self.f_extra_bests_previous:
                next_population.append(deepcopy(population_sorted[ips]))
                
            # do extra mutation with corr for the bests
            if self.f_extra_mutate:
                for j in range(self.n_children):   
                    chromosome = deepcopy(population_sorted[ips]).mutate_ex_crr(self.corr)
                    self.calc_score(chromosome)
                    if chromosome.with_penalty_score > population_sorted[0].with_penalty_score:
                        next_population.append(chromosome)
                
            # mutate the best with minimal chromosome of the best
            if self.f_extra_crossover_with_minimal:
                cols = X.columns[population_sorted[ips].genom]
                b0 = LogisticRegression(penalty='l1').fit(X[cols],y).coef_[0]>0
                min_cols = [ cols[ci] for ci in range(len(cols)) if b0[ci]]
                g0 = np.array([ 'c'+str(i) in min_cols for i in range(1970) ])
                ch0 = _chromosome(self.n_features)
                ch0.create_chromosome_by_genom(g0)
                chromosome = _chromosome(self.n_features)
                chromosome.crossover(population_sorted[ips],ch0)
                self.calc_score(chromosome)
                #if chromosome.with_penalty_score > population_sorted[0].with_penalty_score:
                next_population.append(chromosome)    
                
                
        self.time[-1]['extra_append'] = time()-t0
        return next_population
    
    def keep_history(self,population_sorted,scores_sorted):
        self.verbose('keep history')
        t0 = time()
        
        # diff_with_previous_best
        if len(self.chromosomes_best)>0:
            self.diff_with_previous_best.append(np.count_nonzero(np.logical_xor(self.chromosomes_best[-1].genom , 
                                                                      population_sorted[0].genom))) 
            ch0 = deepcopy(population_sorted[0])
            ch0.diversity_score=math.inf
            ch0.invert_genom = np.invert(ch0.genom)
            for ib in self.chromosomes_best:
                ch0.update_diversity_score(ib)
            self.diff_with_all_bests.append(ch0.diversity_score)
        self.chromosomes_best.append(deepcopy(population_sorted[0]))
        self.mutation_best.append(population_sorted[0].mutated)
        self.max_p.append(population_sorted[0].max_parent_score)
        self.min_p.append(population_sorted[0].min_parent_score)
        self.scores_best.append(scores_sorted[0])
        self.scores_avg.append(np.mean(scores_sorted))
        try:
            self.mutated_avg.append(np.mean([ch.with_penalty_score for ch in population_sorted  if ch.mutated ]))
        except:
            self.mutated_avg.append(0)
            raise
        self.end_of_generation_time.append(time()-self.t0)
        
        self.time[-1]['keep_history'] = time()-t0
    #########################################################
        
    def generate(self, input_population):
        self.verbose('Start generation')
        
        self.time.append({})
        #fitness
        scores_sorted,population_sorted = self.fitness(input_population[:])
        #select
        selected_population = self.select(deepcopy(population_sorted))
        #crossover
        crossovered_population = self.crossover(deepcopy(selected_population))
        #mutate
        mutate_population = self.mutate(deepcopy(crossovered_population))#,plus_mutate_rate_micro)
        #extra_append
        next_population = self.extra_append(deepcopy(mutate_population),population_sorted)
        #keep_history
        self.keep_history(population_sorted,scores_sorted)
        self.verbose('end generate')
        return next_population[:]

    def plot(self,ax4_type='Time_per_Task'):
        
        if 'fig' not in self.__dict__:
            self.fig = plt.figure(num=None, figsize=(18, 12), dpi=120, facecolor='w', edgecolor='k')
            self.ax = self.fig.add_subplot(221)
            self.ax2 = self.fig.add_subplot(222)
            self.ax3 = self.fig.add_subplot(223)
            self.ax4 = self.fig.add_subplot(224)
            self.ax4_2 = self.ax4.twinx()
            plt.ion()

            self.fig.show()
            self.fig.canvas.draw()
            
            
        gen_len = len(self.scores_best)
        range_gl = range(1,gen_len+1)
        
        
#####################################
        self.ax.clear()
        self.ax.set_title('Best & Avg scores of generation\n & Parents scores of The Best \n & Mutation of The best')        
        self.ax.plot(range_gl,self.scores_best,label='The Best')
        self.ax.plot(range_gl,self.scores_avg, label='Average')
        self.ax.plot(range_gl,self.min_p,label='Min parent')
        self.ax.plot(range_gl,self.max_p,label='Max parent')
        #self.ax.plot(range_gl,self.mutated_avg, label='mutated avg')
#         self.ax.plot([g for g in range_gl if self.max_p[g] is not None],
#                  [self.max_p[g] for g in range_gl if self.max_p[g] is not None],
#                label='Max_parent')
        self.ax.scatter([g+1 for g in range(gen_len) if self.mutation_best[g]],
                   [self.scores_best[g] for g in range(gen_len) if self.mutation_best[g]],
                   label='Had mutation')
        self.ax.legend(loc=4)
        self.ax.set_xlabel('Generation')
        self.ax.set_ylabel('Score')

        
#####################################        
        #self.rest_pop_mutated
        '''
        ax_recent = self.ax2.scatter([rpdi+random.uniform(-.4,.4) for rpdi in self.rest_pop_diversity],
                                     self.rest_pop_scores,
                                     alpha=.5*(gen_len-1)/self.max_iterate+.5,marker='*')
        self.ax2.scatter(np.array(self.rest_pop_diversity)[self.rest_pop_mutated],
                         np.array(self.rest_pop_scores)[self.rest_pop_mutated],
                         alpha=.5*(gen_len-1)/self.max_iterate+.5,marker='.',color = 'b')#ax_recent.get_facecolors()[0])
        self.ax2.set_title('population of each generation')
        self.ax2.set_xlabel('Diversity')
        self.ax2.set_ylabel('Score')
        '''
####################################        
        
        # we add [0] because the first generation hasn't previous one to compare
        self.ax3.plot(range_gl,[0]+self.diff_with_previous_best,label='diff_with_previous_best',color='green')
        self.ax3.plot(range_gl,[0]+self.diff_with_all_bests,label='min_diff_with_all_bests',color='orange')
        
        if gen_len==1:
            self.ax3.set_title('Difference of each best with it\'s previous best')
            self.ax3.set_xlabel('Generation')
            self.ax3.set_ylabel('Number of difference in selected features')
            self.ax3.legend(loc=1)

######################################
        
        if ax4_type=='Time_per_Task':
            self.plt_for_legend=[]
            for dtai in range(self.df_time_acc.shape[1]-1):
                pl = plt.bar(range(self.df_time_acc.shape[0]),
                                              self.df_time.iloc[:,dtai+1],
                                              bottom=self.df_time_acc.iloc[:,dtai],
                                              color='C'+str(dtai))
                self.plt_for_legend.append(pl)
            plt.legend([pi[0] for pi in self.plt_for_legend], self.df_time.columns[1:])
        elif ax4_type=='score_per_time':
            self.ax4.clear()
            lns1 = self.ax4.plot(self.end_of_generation_time,self.scores_best,color='brown',label='Score')
            n_features_of_bests = [sum(ch.genom) for ch in self.chromosomes_best]
            lns2 = self.ax4_2.plot(self.end_of_generation_time,n_features_of_bests,color='pink',label='Number of selected features')

            xmin, xmax = self.ax4.get_xlim()
            ymin, ymax = self.ax4.get_ylim()
            xloc = (xmax-xmin)*.7+xmin
            yloc2 = (ymax-ymin)*.55+ymin
            
            self.ax4.text(xloc, yloc2 , 'Pure Score: '+str(round(self.chromosomes_best[-1].pure_score*100,3)
                                                        )+'%\nFeature number: '+str(n_features_of_bests[-1]),
                         bbox={'facecolor': 'blue', 'alpha': 0.07, 'pad': 10})
            #self.ax4.text(xloc, yloc1, 'Feature number: '+str(n_features_of_bests[-1]))

            self.ax4.set_title('Score & Number of selected features for the best per Run-Time')
            self.ax4.set_xlabel('Run-Time in second')
            self.ax4.set_ylabel('Score')
            self.ax4_2.set_ylabel('Number of Features')

            lns = lns1+lns2
            labs = [l.get_label() for l in lns]
            self.ax4.legend(lns, labs, loc=8)
            
#######################################
        self.fig.canvas.draw()
        plt.show()
        if False:
            plt.savefig('Pictures\\GA'+str(gen_len)+'.png')
    
    def fit(self, X, y):
        self.verbose('start fit')
        self.time=[{}]
        t0 = time()
        self.t0 = time()
        
        
        
        self.scores_best , self.scores_avg , self.repeated , self.unique_cnt , self.mutation_best =[],[],[],[],[]
        self.diff_with_previous_best, self.diff_with_all_bests, self.chromosomes_best, self.min_p, self.max_p  = [],[],[],[],[]
        self.end_of_generation_time ,self.mutated_avg= [],[]
        
        self.dataset = X, y
        self.n_features = X.shape[1]
        self.corr=X.corr()
        
        population = self.initilize()

        i=0
        while (i < self.n_gen or                      # number of generation
               np.std(self.scores_best[-10:])!=0 or   # if best score is growing
               np.std(self.max_p[-9:])!=0 or          # if max of parent is growing
               np.std(self.min_p[-8:])!=0             # if min of parent is growing
               ):   
            self.verbose('i='+str(i))
            self.time[-1]['marginal'] = time()-t0
            self.df_time = pd.DataFrame(self.time)
            self.df_time.insert(0,'start',0)
            self.df_time_acc = self.df_time.apply(lambda r:np.add.accumulate(r.fillna(0)),axis=1)
            
            
            
            population = self.generate(population)#=========================================
            
            
            t0 = time()
            if self.show_plots:
                self.plot('score_per_time')
   
            i+=1
            # end function if it reach max iteration
            if i > self.max_iterate:
                break
            
            # if the genetic algorithm doesn't have progress do RFECV with selected columns to reduce extra features
            if self.f_rfecv:
                if i > self.n_gen and np.std(self.min_p[-7:])==0:
                    cols=X.columns[self.chromosomes_best[-1].genom]
                    rfecv =RFECV(estimator=LogisticRegression(), step=1, cv=StratifiedKFold(10),verbose=0,scoring='f1')
                    rfecv.fit(X[cols], y)
                    rfecv_chr = _chromosome(self.n_features,0)
                    rfecv_chr.genom=np.array([ c in cols[rfecv.support_] for c in X.columns])
                    population.append(rfecv_chr)

    # return the best genom
    @property
    def support_(self):
        return self.the_best.genom
    
    
    
    




if __name__ == '__main__':
    import sys
    sys.path.append('../src/')
    
    
    X = pd.read_csv(r'../../data/data_modified.csv',header=None,prefix='c')
    y = pd.read_csv(r'../../data/labels_modified.csv',header=None)
    y.columns=['CANCER_TYPE']
    y_d = pd.get_dummies(y, columns=['CANCER_TYPE'])
    
    sel = GeneticSelector(estimator= LogisticRegression(),#SVC(kernel='linear', C=0.00005), 
                            n_gen=5, 
                            n_best=2, # small number will not coverge, big number respect to n_rand will stick in local optimum
                            n_rand=10, # small number stick in local optimum,   big number has time cost
                            n_children=2, # big number stick in local optimum,    small number will not converge
                            n_die=2,
                            prc_features_rand_init=.05,
                            initial_size_ratio=10,
                            initial_rate_search_score=3,
                            mutation_rate=0.2,
                            scoring="f1",
                            mutate_rate_micro=0.001,
                            l1=0.005,
                            previous_result=None,
                            n_jobs=-1,
                            max_iterate=10,                    
                            f_diverse=True,                    
                            f_periority_in_crossover=False,     
                            f_extra_best_previous=False,        
                            f_extra_bests_previous=False,      
                            f_extra_mutate=False,              
                            f_extra_crossover_with_minimal=False,
                            f_rfecv=True,
                            f_dynamic_child=True,
                            verbose=True,
                            show_plots=False)
    sel.fit(X, y_d['CANCER_TYPE_0'])        