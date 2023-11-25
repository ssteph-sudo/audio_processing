// Save the model (after training and testing) for loading and deployment
// https://waikato.github.io/weka-wiki/serialization/

import weka.core.SerializationHelper;

public void save_model(Classifier cls, String filename){
    weka.core.SerializationHelper.write(filename, cls);
}