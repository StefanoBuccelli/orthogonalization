# orthogonalization
The goal of this small repository is to understand the concept of orthogonalization of power envelopes. To do that, I tried to replicate results from supplementary figure 2, panels "e" and "f" of the paper: Large-scale cortical correlation structure of spontaneous oscillatory activity by Hipp et al. 2012   

Slides to explain results (work in progress): https://docs.google.com/presentation/d/1eXcSeaBakwt9XWHzp2J4sScpUm2wbJgSt8Z3qiOLgGg/edit?usp=sharing 

In scripts:
- hipp_simulation.m is a single script that creates synthetic signals, performs orthogonalization and all the steps needed to reach the goal (not yet fully achieved)
- orthogonalization.m, simply shows in polar coordinates 2 samples from 2 complex signals (that was the starting point)
- understanding_Rayleigh.m I used the definition of Rayleigh from wikipedia and compared with matlab;

References:
- Hipp, Joerg F., et al. "Large-scale cortical correlation structure of spontaneous oscillatory activity." Nature neuroscience 15.6 (2012): 884. Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3861400/ , supplementary: https://media.nature.com/original/nature-assets/neuro/journal/v15/n6/extref/nn.3101-S1.pdf
- Colclough 2015: A symmetric multivariate leakage correction for MEG connectomes https://www.sciencedirect.com/science/article/pii/S1053811915002670?via%3Dihub#f0015 to check also supplementary figures and github repos.
- Omidvarnia, Amir, et al. "Measuring time-varying information flow in scalp EEG signals: orthogonalized partial directed coherence." IEEE transactions on biomedical engineering 61.3 (2014): 680-693. 
- Colclough 2016, How reliable are MEG resting-state connectivity metrics? https://www.sciencedirect.com/science/article/pii/S1053811916301914#f0005 Comparison of the repeatability of 12 common network estimation methods 
- Palva, Ghost interactions in MEG/EEG source space: A note of caution on inter-areal coupling measures. https://www.sciencedirect.com/science/article/pii/S1053811918301290?via%3Dihub 
- Not sure about this: Marqui: Innovations orthogonalization: a solution to the major pitfalls of EEG/MEG “leakage correction” https://arxiv.org/ftp/arxiv/papers/1708/1708.05931.pdf
- O’Neill,  Measuring electrophysiological connectivity by power envelope correlation: A technical review on MEG methods http://eprints.nottingham.ac.uk/31176/2/FC_in_MEG_accepted.pdf
- Siems 2016, Measuring the cortical correlation structure of spontaneous oscillatory activity with EEG and MEG https://www.sciencedirect.com/science/article/pii/S1053811916000707#! 


Github:
- Here you can find different repositories: https://github.com/OHBA-analysis, such as HMM-MAR that deals, among the other things with orthogonalization. Under examples/, there are scripts demonstrating the analysis conducted for the papers: 
    - Diego Vidaurre, Andrew J. Quinn, Adam P. Baker, David Dupret, Alvaro Tejero-Cantero and Mark W. Woolrich (2016) Spectrally resolved fast transient brain states in electrophysiological data. NeuroImage. Volume 126, Pages 81–95.
    - Diego Vidaurre, Romesh Abeysuriya, Robert Becker, Andrew J. Quinn, F. Alfaro-Almagro, S.M Smith and Mark W. Woolrich (2017) Discovering dynamic brain networks from big data in rest and task. NeuroImage.
    - Diego Vidaurre, S.M. Smith and Mark W. Woolrich (2017). Brain network dynamics are hierarchically organized in time. Proceedings of the National Academy of Sciences of the USA.
    - Diego Vidaurre, Lawrence T. Hunt, Andrew J. Quinn, Benjamin A.E. Hunt, Matthew J. Brookes, Anna C. Nobre and Mark W. Woolrich (2017). Spontaneous cortical activity transiently organises into frequency specific phase-coupling networks. Nature Communications.

