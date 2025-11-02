# MLT

Python 3.8.3 and Pytorch 2.5.1 implementation of the Multiplex Latent Trade-off Model (MLT).
## Description
A multiplex social network captures multiple types of social relations among the same set of people, with each layer representing a distinct type of relationship. Understanding the structure of such systems allows us to identify how social exchanges may be driven by a person's own attributes and actions (independence), the status or resources of others (dependence), and mutual influence between entities (interdependence). Characterizing structure in multiplex networks is challenging, as the distinct layers can reflect different yet complementary roles, with interdependence emerging across multiple scales. Here, we introduce the Multiplex Latent Trade-off Model (MLT), a framework for extracting roles in multiplex social networks that accounts for independence, dependence, and interdependence. MLT defines roles as trade-offs, requiring each node to distribute its source and target roles across layers while simultaneously distributing community memberships within hierarchical, multi-scale structures.

## Installation

### Create a Python 3.8.3 environment with conda

```
conda create -n ${env_name} python=3.8.3  
```

### Activate the environment

```
conda activate ${env_name} 
```

### Please install the required packages

```
pip install -r requirements.txt
```


## Learning embeddings for multiplex directed networks using MLT

**RUN:** &emsp; ```python main.py```


## Reference

[Modeling roles and trade-offs in multiplex networks]((https://arxiv.org/abs/2508.05488)). Nikolaos Nakis, Sune Lehmann, Nicholas A. Christakis, and Morten Mørup, Preprint 25





