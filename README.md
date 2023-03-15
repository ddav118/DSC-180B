# DSC-180B <br>
Predicting Pulmonary Edema Using Deep Learning and Image Segmentation <br>
Team Members: David Davila-Garcia, Marco Morocho, Yash Potdar

Note: All data was deidentified but is not publicly available

├── README.md          <- The top-level README for developers using this project.<br>
├── Final_Report.pdf
├── Final_Poster.pdf
├── models             <- Contains the outputs from trained models: Losses and Test Set Predictions<br>
│   ├── Losses         <- Training and Validation MAE Losses by Epoch.<br>
│   ├── Test Set Preds <- NT-proBNP predictions on test set using the best model (minimize MAE valid loss).<br>
├── 1 - Preprocessing.ipynb                               <- Cleaning the original x-rays + clinical data, excluding rows with missing data/no image available<br>
├── 2 - Transfer Learning Training & Evaluation.ipynb     <- (Not used in project) Provided by UCSD AIDA Lab, shows training of U-Net segmentation model.<br>
├── 3 - Predicting Unannotated.ipynb                      <- Used the U-Net segmentation model from the UCSD AIDA Lab to create binary masks (lungs, heart, clavicles, spinal column) for each radiograph in our dataset. Saved the segmentations to an hdf5 file. <br>
├── 4 - Creating Masks.ipynb                              <- Uses the binary masks created in '3 - Predicting Unannotated.ipynb' to produce the segmentation inputs for our model <br>
├── 5 - CNN Models.ipynb                                  <- Contains all code for training and testing models. <br>
├── model.py           <- Contains modified ResNet152 architectures, extends the Pytorch ResNet152 implementation <br>
├── train.py           <- Contains model training and testing functions; different inputs called for different architectures<br>


