# Readme for Assessment 3
1. Task is to identify action from a dataset of clips consisting of 3 frames.
2. Image size is $64 * 64$ pixels.
3. In EDA.py I analyse the different videos in the dataset.
4. In the Dataset the split distribution is as follows:
    * Train : 7770
    * Val :   2230
    * Test :  3270
5. In EDA.py I create train,test and val.csv files that contain the data_id and the label.
6. In models.py I create two models, 1 LSTM and 1 ANN that fine tune on the extracted features of the data set.
7. In models.py use
    ```console
        python models.py --extract_features --pretrained effnet 
    ```
    * to extract features using Efficient net.
    * For MobileNet, replace effnet with mobilenet and for MobileNetV3Small replace effnet with mobile.
8. To fine tune the extracted features run
    ```console
        python models.py --features effnet --lr 1e-3 --epochs 10 --batch_size 32 --model ANN --callbacks
    ```
    * --features use the specified features
    * --lr, --epochs and --batch_size set the laerning rate, epochs and batch_size
    * --model choose the model architecture. Choices are LSTM and ANN.
    * --callbacks sets the Reduce LR on Plateau and Early stopping callbacks during training
9. Use predict.py to generate the predictions file on the test data.
