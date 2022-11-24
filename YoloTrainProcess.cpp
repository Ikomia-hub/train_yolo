#include <QJsonObject>
#include <QJsonArray>
#include "YoloTrainProcess.h"
#include "IO/CDatasetIO.h"
#include "UtilsTools.hpp"

using namespace boost::python;

//---------------------------//
//----- CYoloTrainParam -----//
//---------------------------//
std::map<QString, QString> _modelConfigFiles =
{
    {"yolov4", "template-yolov4.cfg"},
    {"yolov3", "template-yolov3.cfg"},
    {"tiny_yolov4", "template-yolov4-tiny.cfg"},
    {"tiny_yolov3", "template-yolov3-tiny-prn.cfg"},
    {"enet_b0_yolov3", "template-enet-coco.cfg"}
};

std::map<QString, QString> _modelWeightFiles =
{
    {"yolov4", "yolov4.conv.137"},
    {"yolov3", "darknet53.conv.74"},
    {"tiny_yolov4", "yolov4-tiny.conv.29"},
    {"tiny_yolov3", "yolov3-tiny.conv.11"},
    {"enet_b0_yolov3", "enetb0-coco.conv.132"}
};

std::map<QString, std::string> _modelNames =
{
    {"yolov4", "YOLOv4"},
    {"yolov3", "YOLOv3"},
    {"tiny_yolov4", "Tiny YOLOv4"},
    {"tiny_yolov3", "Tiny YOLOv3"},
    {"enet_b0_yolov3", "EfficientNet B0 YOLOv3"}
};

CYoloTrainParam::CYoloTrainParam() : CWorkflowTaskParam()
{
    auto pluginDir = Utils::Plugin::getCppPath() + "/" + Utils::File::conformName("train_yolo").toStdString() + "/";
    m_cfg["model"] = "tiny_yolov4";
    m_cfg["classes"] = "1";
    m_cfg["batchSize"] = "32";
    m_cfg["epochs"] = "1";
    m_cfg["learningRate"] = "0.001";
    m_cfg["momentum"] = "0.9";
    m_cfg["weightDecay"] = "0.0005";
    m_cfg["inputWidth"] = "416";
    m_cfg["inputHeight"] = "416";
    m_cfg["splitRatio"] = "0.9";
    m_cfg["gpuCount"] = "1";
    m_cfg["subdivision"] = "16";
    m_cfg["autoConfig"] = std::to_string(true);
    m_cfg["configPath"] = "";
    m_cfg["outputPath"] = pluginDir + "data/models";;
}

//----------------------//
//----- CYoloTrain -----//
//----------------------//
CYoloTrain::CYoloTrain() : CMlflowTrainTask()
{
    m_pParam = std::make_shared<CYoloTrainParam>();
    addInput(std::make_shared<CDatasetIO>());
    enableTensorboard(false);
}

CYoloTrain::CYoloTrain(const std::string &name, const std::shared_ptr<CYoloTrainParam> &paramPtr) : CMlflowTrainTask(name)
{
    m_pParam = std::make_shared<CYoloTrainParam>(*paramPtr);
    addInput(std::make_shared<CDatasetIO>());
    enableTensorboard(false);
}

size_t CYoloTrain::getProgressSteps()
{
    // Progress steps count is computed from epochs count
    // and updated epochs count is only available when run() is called
    return 1;
}

void CYoloTrain::run()
{
    auto datasetInputPtr = std::dynamic_pointer_cast<CDatasetIO>(getInput(0));
    if(!datasetInputPtr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid dataset input.", __func__, __FILE__, __LINE__);

    if(datasetInputPtr->isDataAvailable() == false)
        throw CException(CoreExCode::INVALID_PARAMETER, "No data available.", __func__, __FILE__, __LINE__);

    auto paramPtr = std::dynamic_pointer_cast<CYoloTrainParam>(m_pParam);
    if(paramPtr == nullptr)
        throw CException(CoreExCode::INVALID_PARAMETER, "Invalid parameters", __func__, __FILE__, __LINE__);

    // Check model
    auto search = m_modelNames.find(paramPtr->m_cfg["model"]);
    if(search == m_modelNames.end())
    {
        std::string models = "";
        for(auto it=m_modelNames.begin(); it!=m_modelNames.end(); ++it)
            models += *it + ",";

        throw CException(CoreExCode::INVALID_PARAMETER, "Invalid model, available models are: " + models, __func__, __FILE__, __LINE__);
    }

    // Dataset preparation
    prepareData();
    beginTaskRun();
    emit m_signalHandler->doAddSubTotalSteps(std::stoi(paramPtr->m_cfg["epochs"]) - 1);

    // Launch training
    launchTraining();

    emit m_signalHandler->doProgress();
    endTaskRun();
}

void CYoloTrain::stop()
{
    m_bStop = true;
}

void CYoloTrain::prepareData()
{
    auto datasetInputPtr = std::dynamic_pointer_cast<CDatasetIO>(getInput(0));
    if(!datasetInputPtr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid dataset input.", __func__, __FILE__, __LINE__);

    auto paramPtr = std::dynamic_pointer_cast<CYoloTrainParam>(m_pParam);
    if(paramPtr == nullptr)
        throw CException(CoreExCode::INVALID_PARAMETER, "Invalid parameters", __func__, __FILE__, __LINE__);

    //Delete files from previous training
    deleteTrainingFiles();

    std::string pluginDir = Utils::Plugin::getCppPath() + "/" + Utils::File::conformName(QString::fromStdString(m_name)).toStdString() + "/";

    // Serialize dataset information from Python struture of IkDatasetIO
    std::string jsonFile = pluginDir + "data/dataset.json";
    datasetInputPtr->save(jsonFile);

    // Read back the dataset as json
    datasetInputPtr->CDatasetIO::load(jsonFile);
    QJsonDocument json = datasetInputPtr->getJsonDocument();

    // Create dataset text annotation files
    if(datasetInputPtr->getSourceFormat() != "yolo")
        createAnnotationFiles(json);

    // Split train-eval
    splitTrainEval(json, std::stof(paramPtr->m_cfg["splitRatio"]));

    // Create class names file
    createClassNamesFile(json);

    // Create config file (.cfg)
    bool bAutoConfig = std::stoi(paramPtr->m_cfg["autoConfig"]);
    if(bAutoConfig)
        createConfigFile();
    else
        updateParamFromConfigFile();

    // Create global data file given to darknet
    createGlobalDataFile();

    // Update config values
    paramPtr->m_cfg["classes"] = std::to_string(m_classCount);
}

void CYoloTrain::createAnnotationFiles(const QJsonDocument &json) const
{
    QJsonObject root = json.object();
    auto itImages = root.find("images");

    if(itImages == root.end())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Invalid dataset structure.", __func__, __FILE__, __LINE__);

    if(itImages.value().isArray() == false)
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Invalid dataset structure.", __func__, __FILE__, __LINE__);

    auto images = itImages.value().toArray();
    for(auto&& imgRef : images)
    {
        auto img = imgRef.toObject();
        auto imgFile = img["filename"].toString();
        boost::filesystem::path imgPath(imgFile.toStdString());
        std::string txtFilePath = imgPath.parent_path().string() + "/" + imgPath.stem().string() + ".txt";
        QFile txtFile(QString::fromStdString(txtFilePath));

        if(txtFile.open(QFile::WriteOnly | QFile::Text))
        {
            QTextStream stream(&txtFile);
            int width = img["width"].toInt();
            int height = img["height"].toInt();
            auto annotations = img["annotations"].toArray();

            for(auto&& annRef : annotations)
            {
                auto ann = annRef.toObject();
                int id = ann["category_id"].toInt();
                stream << id << " ";
                auto coords = ann["bbox"].toArray();
                double x = coords[0].toDouble();
                double y = coords[1].toDouble();
                double w = coords[2].toDouble();
                double h = coords[3].toDouble();
                stream << (x + w/2.0) / (double)width << " ";
                stream << (y + h/2.0) / (double)height << " ";
                stream << w / (double)width << " ";
                stream << h / (double)height << "\n";
            }
            txtFile.close();
        }
    }
}

void CYoloTrain::createClassNamesFile(const QJsonDocument &json)
{
    std::string pluginDir = Utils::Plugin::getCppPath() + "/" + Utils::File::conformName(QString::fromStdString(m_name)).toStdString() + "/";
    QJsonObject root = json.object();
    auto itMetadata = root.find("metadata");

    if(itMetadata == root.end())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Invalid dataset structure.", __func__, __FILE__, __LINE__);

    // Categories are stored in a dict with an integer id as key and class name as value
    // The ids sequence could be sparse, so we must "fill the gap" with "None" class name.
    auto metadata = itMetadata.value().toObject();
    auto categories = metadata["category_names"].toObject();
    QStringList names;
    QStringList classIds = categories.keys();

    m_classCount = 0;
    for(int i=0; i<classIds.size(); ++i)
        m_classCount = std::max(m_classCount, classIds[i].toInt() + 1);

    for(int i=0; i<m_classCount; ++i)
        names.push_back("None");

    for(auto it=categories.begin(); it!=categories.end(); ++it)
    {
        int id = it.key().toInt();
        names[id] = it.value().toString();
    }

    std::string path = pluginDir + "data/classes.txt";
    QFile classFile(QString::fromStdString(path));

    if(classFile.open(QFile::WriteOnly | QFile::Text) == false)
        throw CException(CoreExCode::INVALID_FILE, "Unable to create file classes.txt", __func__, __FILE__, __LINE__);

    QTextStream stream(&classFile);
    for(int i=0; i<names.size(); ++i)
        stream << names[i] << "\n";

    classFile.close();
}

void CYoloTrain::createGlobalDataFile()
{
    auto paramPtr = std::dynamic_pointer_cast<CYoloTrainParam>(m_pParam);
    QString pluginDir = QString::fromStdString(Utils::Plugin::getCppPath()) + "/" + Utils::File::conformName(QString::fromStdString(m_name)) + "/";
    QString path = pluginDir + "data/training.data";
    QFile file(path);

    if(file.open(QFile::WriteOnly | QFile::Text) == false)
        throw CException(CoreExCode::INVALID_FILE, "Unable to create file classes.txt", __func__, __FILE__, __LINE__);

    m_outputFolder = QString::fromStdString(paramPtr->m_cfg["outputPath"]) + "/" + Utils::File::conformName(QDateTime::currentDateTime().toString(Qt::ISODate));
    Utils::File::createDirectory(m_outputFolder.toStdString());

    QTextStream stream(&file);
    stream << "classes = " << m_classCount << "\n";
    stream << "train = " << pluginDir + "data/train.txt\n";
    stream << "valid = " << pluginDir + "data/eval.txt\n";
    stream << "names = " << pluginDir + "data/classes.txt\n";
    stream << "backup = " << m_outputFolder << "\n";
    stream << "metrics = " << pluginDir + "data/metrics.txt";
}

void CYoloTrain::createConfigFile()
{
    auto paramPtr = std::dynamic_pointer_cast<CYoloTrainParam>(m_pParam);
    int epochs = m_classCount * 2000;
    paramPtr->m_cfg["epochs"] = std::to_string(epochs);
    int burnin = (int)(epochs * 0.05);
    int step1 = (int)(epochs * 0.8);
    int step2 = (int)(epochs * 0.9);
    int filters = (m_classCount + 5) * 3;

    QString pluginDir = QString::fromStdString(Utils::Plugin::getCppPath()) + "/" + Utils::File::conformName(QString::fromStdString(m_name)) + "/";
    QString templatePath = pluginDir + "data/config/" + _modelConfigFiles[QString::fromStdString(paramPtr->m_cfg["model"])];
    QString configPath = pluginDir + "data/config/training.cfg";
    paramPtr->m_cfg["configPath"] = configPath.toStdString();

    QFile templateFile(templatePath);
    if(templateFile.open(QFile::ReadOnly | QFile::Text) == false)
        throw CException(CoreExCode::INVALID_PARAMETER, QObject::tr("Unable to read config template file.").toStdString(), __func__, __FILE__, __LINE__);

    QTextStream txtStream(&templateFile);
    auto templateContent = txtStream.readAll();
    templateFile.close();

    auto newContent = templateContent.replace("_batch_", QString::fromStdString(paramPtr->m_cfg["batchSize"]));
    newContent = newContent.replace("_subdivision_", QString::fromStdString(paramPtr->m_cfg["subdivision"]));
    newContent = newContent.replace("_width_", QString::fromStdString(paramPtr->m_cfg["inputWidth"]));
    newContent = newContent.replace("_height_", QString::fromStdString(paramPtr->m_cfg["inputHeight"]));
    newContent = newContent.replace("_momentum_", QString::fromStdString(paramPtr->m_cfg["momentum"]));
    newContent = newContent.replace("_decay_", QString::fromStdString(paramPtr->m_cfg["weightDecay"]));
    newContent = newContent.replace("_lr_", QString::fromStdString(paramPtr->m_cfg["learningRate"]));
    newContent = newContent.replace("_burnin_", QString::number(burnin));
    newContent = newContent.replace("_epochs_", QString::fromStdString(paramPtr->m_cfg["epochs"]));
    newContent = newContent.replace("_steps_", QString::number(step1)+ "," + QString::number(step2));
    newContent = newContent.replace("_filters_", QString::number(filters));
    newContent = newContent.replace("_classes_", QString::number(m_classCount));

    QFile configFile(configPath);
    if(configFile.open(QFile::WriteOnly | QFile::Text | QFile::Truncate) == false)
        throw CException(CoreExCode::INVALID_PARAMETER, QObject::tr("Unable to write auto-generated config file.").toStdString(), __func__, __FILE__, __LINE__);

    QTextStream outTextStream(&configFile);
    outTextStream << newContent;
    configFile.close();
}

void CYoloTrain::updateParamFromConfigFile()
{
    auto paramPtr = std::dynamic_pointer_cast<CYoloTrainParam>(m_pParam);
    QFile configFile(QString::fromStdString(paramPtr->m_cfg["configPath"]));

    if(configFile.open(QFile::ReadOnly | QFile::Text) == false)
        return;

    QTextStream stream(&configFile);
    QString config = stream.readAll();

    //Parse config file
    QRegularExpression re;
    QRegularExpressionMatch match;

    //Batch
    re.setPattern("batch *= *([0-9]+)");
    match = re.match(config);

    if(match.hasMatch())
        paramPtr->m_cfg["batchSize"] = match.captured(1).toStdString();

    //Subdivision
    re.setPattern("subdivisions *= *([0-9]+)");
    match = re.match(config);

    if(match.hasMatch())
        paramPtr->m_cfg["subdivision"] = match.captured(1).toStdString();

    //Input width
    re.setPattern("width *= *([0-9]+)");
    match = re.match(config);

    if(match.hasMatch())
        paramPtr->m_cfg["inputWidth"] = match.captured(1).toStdString();

    //Input height
    re.setPattern("width *= *([0-9]+)");
    match = re.match(config);

    if(match.hasMatch())
        paramPtr->m_cfg["inputHeight"] = match.captured(1).toStdString();

    //Momentum
    re.setPattern("momentum *= *([0-9.]+)");
    match = re.match(config);

    if(match.hasMatch())
        paramPtr->m_cfg["momentum"] = match.captured(1).toStdString();

    //Decay
    re.setPattern("decay *= *([0-9.]+)");
    match = re.match(config);

    if(match.hasMatch())
        paramPtr->m_cfg["weightDecay"] = match.captured(1).toStdString();

    //Learning rate
    re.setPattern("learning_rate *= *([0-9.]+)");
    match = re.match(config);

    if(match.hasMatch())
        paramPtr->m_cfg["learningRate"] = match.captured(1).toStdString();

    //Epochs
    re.setPattern("max_batches *= *([0-9]+)");
    match = re.match(config);

    if(match.hasMatch())
        paramPtr->m_cfg["epochs"] = match.captured(1).toStdString();
}

void CYoloTrain::splitTrainEval(const QJsonDocument &json, float ratio)
{
    std::string pluginDir = Utils::Plugin::getCppPath() + "/" + Utils::File::conformName(QString::fromStdString(m_name)).toStdString() + "/";
    QJsonObject root = json.object();
    auto itImages = root.find("images");

    if(itImages == root.end())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Invalid dataset structure.", __func__, __FILE__, __LINE__);

    if(itImages.value().isArray() == false)
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Invalid dataset structure.", __func__, __FILE__, __LINE__);

    std::vector<std::string> imagePaths;
    auto images = itImages.value().toArray();

    for(auto&& imgRef : images)
    {
        auto img = imgRef.toObject();
        imagePaths.push_back(img["filename"].toString().toStdString());
    }

    // Split dataset randomly
    size_t trainSize = (size_t)(ratio * imagePaths.size());
    std::random_shuffle(imagePaths.begin(), imagePaths.end());
    auto trainImgPaths = std::vector<std::string>(imagePaths.begin(), imagePaths.begin() + trainSize);
    auto evalImgPaths = std::vector<std::string>(imagePaths.begin() + trainSize, imagePaths.end());

    // Save file train.txt containing image paths of training set
    std::string trainPath = pluginDir + "data/train.txt";
    QFile trainFile(QString::fromStdString(trainPath));

    if(trainFile.open(QFile::WriteOnly | QFile::Text) == false)
        throw CException(CoreExCode::INVALID_FILE, "Unable to create file train.txt", __func__, __FILE__, __LINE__);

    QTextStream trainStream(&trainFile);
    for(size_t i=0; i<trainImgPaths.size(); ++i)
        trainStream << QString::fromStdString(trainImgPaths[i]) << "\n";

    trainFile.close();

    // Save file eval.txt containing image paths of evaluation set
    std::string evalPath = pluginDir + "data/eval.txt";
    QFile evalFile(QString::fromStdString(evalPath));

    if(evalFile.open(QFile::WriteOnly | QFile::Text) == false)
        throw CException(CoreExCode::INVALID_FILE, "Unable to create file eval.txt", __func__, __FILE__, __LINE__);

    QTextStream evalStream(&evalFile);
    for(size_t i=0; i<evalImgPaths.size(); ++i)
        evalStream << QString::fromStdString(evalImgPaths[i]) << "\n";

    evalFile.close();
}

void CYoloTrain::launchTraining()
{
    auto paramPtr = std::dynamic_pointer_cast<CYoloTrainParam>(m_pParam);
    QString pluginDir = QString::fromStdString(Utils::Plugin::getCppPath()) + "/" + Utils::File::conformName(QString::fromStdString(m_name)) + "/";
    QString dataFilePath = pluginDir + "data/training.data";
    QString configFilePath = QString::fromStdString(paramPtr->m_cfg["configPath"]);
    QString weightsFilePath = pluginDir + "data/models/pretrained/" + _modelWeightFiles[QString::fromStdString(paramPtr->m_cfg["model"])];
    QString metricsFilePath = pluginDir + "data/metrics.txt";
    QString logFilePath = pluginDir + "data/log.txt";
    QString darknetExe = pluginDir + "darknet";

    std::string weightPath = weightsFilePath.toStdString();
    if (!Utils::File::isFileExist(weightPath))
    {
        std::cout << "Downloading model..." << std::endl;
        std::string modelName = _modelWeightFiles[QString::fromStdString(paramPtr->m_cfg["model"])].toStdString();
        std::string downloadUrl = Utils::Plugin::getModelHubUrl() + "/" + m_name + "/" + modelName;
        download(downloadUrl, weightPath);
    }

    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();

    auto libFolder = QString::fromStdString(Utils::IkomiaApp::getIkomiaLibFolder());
    if(QDir(libFolder).exists())
    {

#if defined(Q_OS_WIN64)
#elif defined(Q_OS_LINUX)
        QString libPath = env.value("LD_LIBRARY_PATH");
        if(!libPath.contains(libFolder))
        {
            libPath = libFolder + ":" + libPath;
            env.insert("LD_LIBRARY_PATH", libPath);
        }
#elif defined(Q_OS_MACOS)
#endif
    }

    QStringList args;
    args << "detector" << "train" << dataFilePath << configFilePath << weightsFilePath << "-dont_show" << "-map" << "-log_metrics";

    QProcess proc;
    proc.setProcessEnvironment(env);
    proc.setProcessChannelMode(QProcess::MergedChannels);
    proc.setStandardOutputFile(logFilePath, QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text);
    proc.start(darknetExe, args);
    proc.waitForStarted();

    QFile metricsFile(metricsFilePath);
    while(!metricsFile.exists(metricsFilePath));
    metricsFile.open(QFile::ReadOnly | QFile::Text);
    QTextStream metricsStream(&metricsFile);

    //MLflow is quiet slow, we log metrics asynchronously
    m_mlflowLogFreq = std::max(1, std::stoi(paramPtr->m_cfg["epochs"]) / 100);
    auto mlflowFuture = Utils::async([&]
    {
        while(!m_metricsQueue.empty() || m_bFinished == false)
        {
            if(!m_metricsQueue.empty())
            {
                auto metrics = m_metricsQueue.front();
                m_metricsQueue.pop();
                int epoch = (int)metrics["Epoch"];
                metrics.erase("Epoch");
                logMetrics(metrics, epoch - 1);
            }
        }
    });

    while(!proc.waitForFinished(1) && m_bStop == false)
        loadMetrics(metricsStream);

    if(m_bStop)
    {
        proc.kill();
        m_bStop = false;
    }
    else
    {
        auto status = proc.exitStatus();
        if(status == QProcess::CrashExit)
            throw CException(CoreExCode::UNKNOWN, "Darknet internal error.");
    }
    m_bFinished = true;

    //Wait for MLflow logging process - timeout: 2 min
    emit m_signalHandler->doLog("Waiting for MLflow logging process...");
    mlflowFuture.wait_for(std::chrono::minutes(2));

    //Log config file
    logArtifact(configFilePath.toStdString());

    //Copy files needed for inference
    auto outFolder = m_outputFolder.toStdString();
    auto pluginFolder = pluginDir.toStdString();
    boost::filesystem::copy_file(pluginFolder + "data/config/training.cfg", outFolder + "/training.cfg", boost::filesystem::copy_option::overwrite_if_exists);
    boost::filesystem::copy_file(pluginFolder + "data/classes.txt", outFolder + "/classes.txt", boost::filesystem::copy_option::overwrite_if_exists);

    emit m_signalHandler->doLog("YOLO training finished!");
}

void CYoloTrain::loadMetrics(QTextStream& stream)
{
    YoloMetrics metrics;
    std::vector<std::string> values;

    if(stream.atEnd())
        return;

    auto str = stream.readLine();
    Utils::String::tokenize(str.toStdString(), values, " ");
    if(values.size() != 4)
        return;

    int iteration = std::stoi(values[0]);
    metrics["Epoch"] = iteration;
    metrics["Loss"] = std::stof(values[1]);
    metrics["mAP"] = std::stof(values[2]);
    metrics["Best mAP"] = std::stof(values[3]);

    auto logMsg = QString("Epoch #%1 - Loss = %2 - mAP = %3 - Best mAP = %4")
            .arg(QString::fromStdString(values[0]))
            .arg(QString::fromStdString(values[1]))
            .arg(QString::fromStdString(values[2]))
            .arg(QString::fromStdString(values[3]));

    emit m_signalHandler->doLog(logMsg);
    emit m_signalHandler->doProgress();

    if(iteration % m_mlflowLogFreq == 1)
        m_metricsQueue.push(metrics);
}

void CYoloTrain::deleteTrainingFiles()
{
    QString dataDir = QString::fromStdString(Utils::Plugin::getCppPath()) + "/" + Utils::File::conformName(QString::fromStdString(m_name)) + "/data/";
    QFile::remove(dataDir + "log.txt");
    QFile::remove(dataDir + "metrics.txt");
}

