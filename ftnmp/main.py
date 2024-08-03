import torch as tc
import ftnmp.graph_generate as gg
import numpy as np
import ftnmp.contract_exact as ce
import time
import ftnmp.EntropyBP as EBP
import ftnmp.EntropyFTNMP as EFT
import ftnmp.get_regions_new as grn
from tqdm import tqdm




device = 'cuda'
R_subregion = 100
error_FTNBP_6=[]
error_BP = []
file = open('FTNMP.txt','w')
times = []
Error =[]
Entropy =[]
times =[]
steps =[]
seed =0
oom = 0
for i in tqdm(range(0,50)):
    t = []
    Ey = []
    Er = []
    step=[]
    nodes = 300
    alpha =0.7
    n_cl = 0
    alpha_local=[2.5,3.2]
    max_cl = 12
    min_cl = 8
    clauses,tendencies = gg.double_random_generate(nodes,alpha,n_cl,alpha_local,min_cl,max_cl,seed)
    print('nodes:',nodes,'alpha_grobal:',alpha,'number of cluster:',n_cl,'alpha_local:',alpha_local,'Maximum points:',max_cl,'Minimum points:',min_cl,file = file)
    print('clauses=',clauses,file =file)
    print('tendencies=',tendencies,file = file)
    print('alpha:',len(clauses)/nodes,file = file)
    G_fac,max_item =gg.G_generator_fac(clauses)
    nodes = list(range(max_item+1))
    print(f"Step {i}  begin:")
    try:
       
        configrite_number_qu,sgn,T =ce.local_contraction_ds(G_fac,clauses,tendencies,max_item,25,device)
        entropy_exact_qu = tc.log(configrite_number_qu)+sgn*tc.log(tc.tensor(2))
        entropy_exact_qu = tc.where(entropy_exact_qu.isnan() | entropy_exact_qu.isneginf(), tc.tensor(0.0).to(device), entropy_exact_qu)
        print( 'time_opt:',T,entropy_exact_qu)
        t.append(T)
        if entropy_exact_qu !=0:
            Ey.append(entropy_exact_qu.item())
            Er.append(0)
        else:
            continue

        print('\n','BP:')
        t5 = time.time()
        entropy_BP,step_BP = EBP.EntropyBP(clauses,tendencies)
        entropy_BP = tc.nan_to_num(entropy_BP, nan=0.0)
        t6 = time.time()
        print('BP   :',entropy_BP)
        t.append(t6-t5)
        step.append(step_BP)
        if entropy_exact_qu !=0:
            Ey.append(entropy_BP.item())
            Er.append((abs(entropy_BP-entropy_exact_qu)/entropy_exact_qu).item())
        
       
        R_region = 4
        devides,subdevides,region_info,single_list = grn.get_regions_z(G_fac,clauses,max_item,R_region,R_subregion)
        Nv,boundaries = grn.devides_to_Nv(G_fac,devides,region_info)
        devide_bound,degree_node = grn.devides_others(G_fac,devides)
        print('\n','FTNBP_4:')
        t7 = time.time()
        entropy_FTNBP_4,step_4 =EFT.EntropyZ_new(G_fac,clauses,tendencies,devides,devide_bound,region_info,single_list,max_item,Nv,boundaries,degree_node,device,file)
        entropy_FTNBP_4 = tc.nan_to_num(entropy_FTNBP_4, nan=0.0)
        t8 = time.time()
        print('FTNBP_4:',entropy_FTNBP_4)
        t.append(t8-t7)
        step.append(step_4)
        if entropy_exact_qu!=0:
            Ey.append(entropy_FTNBP_4.item())
            Er.append((abs(entropy_FTNBP_4-entropy_exact_qu)/entropy_exact_qu).item())
        

        R_region = 6
        devides,subdevides,region_info,single_list = grn.get_regions_z(G_fac,clauses,max_item,R_region,R_subregion)
        Nv,boundaries = grn.devides_to_Nv(G_fac,devides,region_info) 
        devide_bound,degree_node = grn.devides_others(G_fac,devides)
        print('\n','FTNBP_6:')
        t3 = time.time()
        entropy_FTNBP_6,step_6 =EFT.EntropyZ_new(G_fac,clauses,tendencies,devides,devide_bound,region_info,single_list,max_item,Nv,boundaries,degree_node,device,file)
        entropy_FTNBP_6 = tc.nan_to_num(entropy_FTNBP_6, nan=0.0)
        t4 = time.time()
        print('FTNBP_6:',entropy_FTNBP_6)
        t.append(t4-t3)
        step.append(step_6)
        if entropy_exact_qu!=0:
            Ey.append(entropy_FTNBP_6.item())
            Er.append((abs(entropy_FTNBP_6-entropy_exact_qu)/entropy_exact_qu).item())
        

        
        

        print(f"Step {i} information as follow:")
        print('exact  :',entropy_exact_qu)
        print('BP     :',entropy_BP)
        print('FTNBP_4:',entropy_FTNBP_4)
        print('FTNBP_6:',entropy_FTNBP_6)
        print('error_BP   :',abs(entropy_exact_qu - entropy_BP))
        print('error_FTNBP_4:',abs(entropy_exact_qu - entropy_FTNBP_4))
        print('error_FTNBP_6:',abs(entropy_exact_qu - entropy_FTNBP_6))
        
        
        
        times.append(t)
        steps.append(step)
        if len(Ey)!=0:
            Entropy.append(Ey)
            Error.append(Er)
       
        Err = np.array(Error)
        column_means_Er = np.mean(Err, axis=0)
      

        



        
        print('exact  :',entropy_exact_qu,file = file)
        print('BP     :',entropy_BP,file = file)
        print('FTNBP_4:',entropy_FTNBP_4,file = file)
        print('FTNBP_6:',entropy_FTNBP_6,file = file)
        print('error_BP   :',abs(entropy_exact_qu - entropy_BP)/entropy_exact_qu,file = file)
        print('error_FTNBP_4:',abs(entropy_exact_qu - entropy_FTNBP_4)/entropy_exact_qu,file=file )
        print('error_FTNBP_6:',abs(entropy_exact_qu - entropy_FTNBP_6)/entropy_exact_qu,file = file )
        print('avr_error_BP     :',column_means_Er[1],file = file)
        print('avr_error_FTNBP_4:',column_means_Er[2],file = file )
        print('avr_error_FTNBP_6:',column_means_Er[3],file = file )

        print('_____________________________________ERROR_________________________________________',file=file)
        Err = np.array(Error)
        print(Error,file = file)
        column_means_Er = np.mean(Err, axis=0)
        print('EXACT---BP---E4---E6----Errorbar:', '\n',column_means_Er,file=file)
        max_valuesr = np.max(Err, axis=0)
        min_valuesr = np.min(Err, axis=0)
        resultr = list(zip(min_valuesr.reshape(1, -1)[0], max_valuesr.reshape(1, -1)[0]))
        print('EXACT---BP---E4---E6----Error(min,max):', '\n',resultr,file=file)
        print('_________________________________instance error____________________________________',file=file)
        print(Entropy,file=file)
        E = np.array(Entropy)
        column_means_E = np.mean(E, axis=0)
        print('EXACT---BP---E4---E6----Sbar:', '\n',column_means_E,file=file)
        max_values = np.max(E, axis=0)
        min_values = np.min(E, axis=0)
        result = list(zip(min_values.reshape(1, -1)[0], max_values.reshape(1, -1)[0]))
        print('EXACT---BP---E4---E6----S(min,max):', '\n',result,file=file)
        print('_________________________________average time_______________________________________',file=file)
        print(times,file = file)
        T = np.array(times)
        column_means = np.mean(T, axis=0)
        print('EXACT---BP---E4---E6----Tbar:', '\n',column_means,file=file)
        max_valuest = np.max(T, axis=0)
        min_valuest = np.min(T, axis=0)
        resultt = list(zip(min_valuest.reshape(1, -1)[0], max_valuest.reshape(1, -1)[0]))
        print('EXACT---BP---E4---E6----T(min,max):', '\n',resultt,file=file)
        print('_______________________________________steps_______________________________________',file=file)
        print(steps,file = file)
        STEP  = np.array(steps)
        column_means_steps = np.mean(STEP,axis=0)
        print('BP---E4---E6----Stepsbar:', '\n',column_means_steps,file=file)
        max_valuesp = np.max(STEP, axis=0)
        min_valuesp = np.min(STEP, axis=0)
        resultp= list(zip(min_valuesp.reshape(1, -1)[0], max_valuesp.reshape(1, -1)[0]))
        print('BP---E4---E6----Steps(min,max):', '\n',resultp,file=file)
        print('\n','\n',file = file)
        print(f"Step {i} completed successfully.")
    except RuntimeError as e:
        if 'out of memory' in str(e):
            oom = oom+1
            print(f"Step {i} skipped due to OOM error.")
        raise 
    seed = seed + 1
print('oom:',oom,file = file)



