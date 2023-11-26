// Modified from https://waikato.github.io/weka-wiki/use_weka_in_your_java_code/#datasets
// Read the processed features from a csv file
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.instance;
 
// Provide a file path to a csv file of features
public Instances read_training_data(csv_file_path){
    DataSource source = new DataSource(csv_file_path);
    Instances data = source.getDataSet();

    // setting class attribute if the data format does not provide this information
    // For example, the XRFF format saves the class attribute information as well
    if (data.classIndex() == -1)
        data.setClassIndex(data.numAttributes() - 1);

    return data;
}
