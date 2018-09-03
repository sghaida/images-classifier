### Image classification project
the project is intended to classify flowers based on their images

#### usage

to train neural network:
```commandline
python train.py --data_dir "flowers" --gpu --epochs 5 --arch "vgg16" --learning_rate 0.001 --hidden_units 3136 784  --save_dir "models"
```

to predict
```commandline
python predict.py --image "flowers/valid/1/image_06739.jpg" --topk 6 --labels "cat_to_name.json" --checkpoint "models/checkpoint.pth"
```

#### notes
the source of the images is available [here](https://view0228105f.udacity-student-workspaces.com/file_download/tmp/0.8389386249711195/flowers.tar.gz) 
please download, extract them and put them in the project path flowes. 