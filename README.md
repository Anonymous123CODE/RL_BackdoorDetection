# RL_BackdoorDetection

# Dependencies 

Our code is implemented using Keras with TensorFlow backend. Following packages are required.


  *keras==2.4

  *tensorflow-gpu==2.6

  *mujoco = 1.31

  *Our code is tested on Python 3.8.3

To install the Mujoco environment you can refer:

```bash
https://github.com/openai/multiagent-competition
```
Notably,  we change the environment by :

```bash
env = gym.make("multicomp/RunToGoalHumans-v0")
```
For Mujoco with different versions, you can also change the environment by:

```bash
env = gym.make("run-to-goal-humans-v0")
```

We include a sample script demonstrating how to perform PolicyCleanse on a target RL model. There are several parameters that need to be set before running the code:


On GPU device: 
if you are using GPU, specify which GPU you would like to use by setting the DEVICE variable via
```bash
python run.py --cuda 0
```

On Seed of environment:
you can set the seed of environment via

```bash
python run.py --seed 111
```

If you want to test the code on your own models, please specify the path to the model and corresponding parameters varialble in run.py & ppo.py .


For example, you can access Run-To-Goal(Ants) model throuhg:
```bash 
import tensorflow.keras as keras
model=keras.models.load_model("saved_models/trojnn_incre.h5")
```
The parameters for Run-To-Goals is stored at 
```bash
parameters/ants_to_go/
```
To obtain the detection results for Run-To-Goal(Humans) using PolicyCleanse, you can run:

```bash
python run.sh --cuda 0 --seed 111
```
You can also test on models for other games stored in saved_models package:

```bash
cd saved_models/
```

We also include the videos for BackdooRL in videos directory


The output is shown as below:

```bash 
....
Total Reward:3.280012822397361
Rollouts: 7
Total Reward:86.49874293325423
Rollouts: 8
Total Reward:-89.78765717885204
Rollouts: 9
GG:32.46787860775251
Total Reward:3.280012822397361
Rollouts: 10
Total Reward:79.00201563912883
Rollouts: 11
Total Reward:38.89453083931335
Rollouts: 12
Total Reward:71.36923693771
Rollouts: 13
Total Reward:11.30939614406186
Rollouts: 14
Total Reward:-7.520163000338677
Rollouts: 15
Total Reward:-84.4008571071478
Rollouts: 16
Total Reward:-84.50891597264538
Rollouts: 17
Total Reward:67.29059632333626
Rollouts: 18
Total Reward:201.86346226737976
Rollouts: 19
Total Reward:-74.86755748781822
Rollouts: 20
Total Reward:-40.13878301975884
Rollouts: 21
Total Reward:-91.12859993801723
Rollouts: 22
Total Reward:-47.45661990733305
Rollouts: 23
Total Reward:74.60566298595383
Rollouts: 24
Total Reward:106.02855498894
Rollouts: 25
Total Reward:56.23331582588901
Rollouts: 26
Total Reward:-75.6978408452654
Rollouts: 27
Total Reward:-65.25007361989086
Rollouts: 28
Total Reward:17.703876678644484
Rollouts: 29
Total Reward:-15.371445289553535
Rollouts: 30
Total Reward:69.0139169301308
Rollouts: 31
Total Reward:-75.26812544825279
Rollouts: 32
Total Reward:-84.38799666096982
Rollouts: 33
Total Reward:32.04490074913728
Rollouts: 34
Total Reward:-48.0324199538409
Rollouts: 35
Total Reward:-51.26293263997297
Rollouts: 36
Total Reward:-89.5110497640782
Rollouts: 37
Total Reward:72.85692108134973
Rollouts: 38
Total Reward:47.530317929633014
Rollouts: 39
Total Reward:71.2609890665381
Rollouts: 40
Total Reward:-85.20078918164926
Rollouts: 41
Total Reward:-62.265095056436145
Rollouts: 42
Total Reward:109.77318071501732
Rollouts: 43
Total Reward:105.59494314654292
Rollouts: 44
Total Reward:85.6243757710173
Rollouts: 45
Total Reward:-71.75208364288508
Rollouts: 46
Total Reward:-86.63331785933745
Rollouts: 47
Total Reward:-1.0664511128821301
Rollouts: 48
Total Reward:64.89283570505232
Rollouts: 49
Total Reward:-63.79930197534812
Rollouts: 50
Total Reward:-16.58129565917156
Rollouts: 51
Total Reward:70.20964323648617
Rollouts: 52
Total Reward:52.24469159162997
Rollouts: 53
Total Reward:-74.01710225544107
Rollouts: 54
Total Reward:36.4174166907423
Rollouts: 55
Total Reward:69.07026534797447
Rollouts: 56
Total Reward:-70.19248625435425
Rollouts: 57
Total Reward:87.34013659160975
Rollouts: 58
Total Reward:-5.062542963285921
Rollouts: 59
Total Reward:23.961750186386926
Rollouts: 60
Total Reward:132.39415874206455
Rollouts: 61
Total Reward:-25.343440639821626
Rollouts: 62
Total Reward:52.552681158460786
Rollouts: 63
Total Reward:79.46727063778053
Rollouts: 64
Total Reward:86.08575139210969
Rollouts: 65
Total Reward:150.24996706790665
Rollouts: 66
Total Reward:157.64279188168564
Rollouts: 67
Total Reward:-17.473249780471775
Rollouts: 68
Total Reward:-45.07573820780698
Rollouts: 69
Total Reward:84.97945920318156
Rollouts: 70
Total Reward:171.66702469979904
Rollouts: 71
Total Reward:7.9327764265213006
Rollouts: 72
Total Reward:-84.37229080151182
Rollouts: 73
Total Reward:-80.74349365781026
Rollouts: 74
Total Reward:41.930285780273856
Rollouts: 75
Total Reward:107.45211966588958
Rollouts: 76
Total Reward:76.53829161206833
Rollouts: 77
Total Reward:130.80150648841092
Rollouts: 78
Total Reward:-74.25600425725123
Rollouts: 79
Total Reward:122.64099266557749
Rollouts: 80
Total Reward:-30.188399779617523
Rollouts: 81
Total Reward:43.82370140925774
Rollouts: 82
Total Reward:-91.98036227497272
Rollouts: 83
Total Reward:79.23998493230113
Rollouts: 84
Total Reward:23.083203205239368
Rollouts: 85
Total Reward:-30.9724996882982
Rollouts: 86
Total Reward:54.1234808187709
Rollouts: 87
Total Reward:65.53846986302976
Rollouts: 88
Total Reward:-80.34037903906892
Rollouts: 89
Total Reward:62.37264765687752
Rollouts: 90
Total Reward:-85.7996599503365
Rollouts: 91
Total Reward:-31.834475441472765
Rollouts: 92
Total Reward:103.7641030881394
Rollouts: 93
Total Reward:-82.24930708712554
Rollouts: 94
Total Reward:90.54569889653791
Rollouts: 95
Total Reward:203.5013255526103
Epochs:95 End(Finds Triggers)...
```


