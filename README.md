# Social-RippleNet

## Social-RippleNet: Jointly Modeling of Ripple Net and Social Information for Recommendation


Social-RippleNet consists of user modeling, item modeling, and rating prediction. 

  - User modeling: User modeling combines user interaction features, user social features, and user-own features to obtain user latent features. 

  - Item modeling: Item modeling combines item interaction features, item knowledge graph features, and item-own features to obtain item latent features, wherein the aggregation of knowledge graph based on the idea of multiple “ripples” superposition to improve Ripple Net for the extraction of item knowledge graph features. 

  - Rating prediction: Rating prediction combines user and item latent features to predict the user rating of items. 

 
 ## Code
The code is modified on the basis of Ripple Net and GraphRec:

  - For the authors' official TensorFlow implementation, see [hwwang55/RippleNet](https://github.com/hwwang55/RippleNet).
  - GraphRec Author: Wenqi Fan (https://wenqifan03.github.io, email: wenqifan03@gmail.com)



### Files in the folder


  - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
  - `kg_part1.txt` and `kg_part2.txt`: knowledge graph file;
  - `ratrings.dat`: raw rating file of MovieLens-1M
  - `movies.dat`: raw movies file of MovieLens-1M
  - `users.dat`: raw users file of MovieLens-1M
  - `350000output.txt` and `500000output.txt`: Data obtained after data preprocessing






### Required packages
The code has been tested running under Python 3.6, with the following packages installed (along with their dependencies):
- pytorch >= 1.0
- numpy >= 1.14.5
- sklearn >= 0.19.1


### Running the code
```
python preprocess.py 
python run_Social-RippleNet.py 
```
