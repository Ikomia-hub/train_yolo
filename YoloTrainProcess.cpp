#include <QJsonObject>
#include <QJsonArray>
#include "YoloTrainProcess.h"
#include "IO/CDatasetIO.hpp"
#include "UtilsTools.hpp"

using namespace boost::python;

std::map<int, QString> _modelConfigFiles =
{
    {CYoloTrainParam::YOLOV4, "template-yolov4.cfg"},
    {CYoloTrainParam::YOLOV3, "template-yolov3.cfg"},
    {CYoloTrainParam::TINY_YOLOV4, "template-yolov4-tiny.cfg"},
    {CYoloTrainParam::TINY_YOLOV3, "template-yolov3-tiny-prn.cfg"},
    {CYoloTrainParam::ENET_B0_YOLOV3, "template-enet-coco.cfg"}
};

std::map<int, QString> _modelWeightFiles =
{
    {CYoloTrainParam::YOLOV4, "yolov4.conv.137"},
    {CYoloTrainParam::YOLOV3, "darknet53.conv.74"},
    {CYoloTrainParam::TINY_YOLOV4, "yolov4-tiny.conv.29"},
    {CYoloTrainParam::TINY_YOLOV3, "yolov3-tiny.conv.11"},
    {CYoloTrainParam::ENET_B0_YOLOV3, "enetb0-coco.conv.132"}
};

std::map<int, std::string> _modelNames =
{
    {CYoloTrainParam::YOLOV4, "YOLOv4"},
    {CYoloTrainParam::YOLOV3, "YOLOv3"},
    {CYoloTrainParam::TINY_YOLOV4, "Tiny YOLOv4"},
    {CYoloTrainParam::TINY_YOLOV3, "Tiny YOLOv3"},
    {CYoloTrainParam::ENET_B0_YOLOV3, "EfficientNet B0 YOLOv3"}
};

CYoloTrain::CYoloTrain() : CMlflowTrainProcess()
{
    m_pParam = std::make_shared<CYoloTrainParam>();
    addInput(std::make_shared<CDatasetIO>());
}

CYoloTrain::CYoloTrain(const std::string &name, const std::shared_ptr<CYoloTrainParam> &paramPtr) : CMlflowTrainProcess(name)
{
    m_pParam = std::make_shared<CYoloTrainParam>(*paramPtr);
    addInput(std::make_shared<CDatasetIO>());
}

size_t CYoloTrain::getProgressSteps()
{
    auto paramPtr = std::dynamic_pointer_cast<CYoloTrainParam>(m_pParam);
    prepareData();
    return paramPtr->m_epochs;
}

void CYoloTrain::run()
{
    beginTaskRun();

    auto datasetInputPtr = std::dynamic_pointer_cast<CDatasetIO>(getInput(0));
    if(!datasetInputPtr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid dataset input.", __func__, __FILE__, __LINE__);

    if(datasetInputPtr->isDataAvailable() == false)
        throw CException(CoreExCode::INVALID_PARAMETER, "No data available.", __func__, __FILE__, __LINE__);

    auto paramPtr = std::dynamic_pointer_cast<CYoloTrainParam>(m_pParam);
    if(paramPtr == nullptr)
        throw CException(CoreExCode::INVALID_PARAMETER, "Invalid parameters", __func__, __FILE__, __LINE__);

    // MLflow parameters logging
    initParamsLogging();

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
    if(datasetInputPtr->load(jsonFile) == false)
        throw CException(CoreExCode::INVALID_FILE, "Invalid dataset JSON file.", __func__, __FILE__, __LINE__);

    QJsonDocument json = datasetInputPtr->getJsonDocument();

    // Create dataset text annotation files
    if(datasetInputPtr->getSourceFormat() != CDatasetIO::Format::YOLO)
        createAnnotationFiles(json);

    // Split train-eval
    splitTrainEval(json, paramPtr->m_splitRatio);

    // Create class names file
    createClassNamesFile(json);

    // Create config file (.cfg)
    if(paramPtr->m_bAutoConfig)
        createConfigFile();
    else
        updateParamFromConfigFile();

    // Create global data file given to darknet
    createGlobalDataFile();
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
    QString pluginDir = QString::fromStdString(Utils::Plugin::getCppPath()) + "/" + Utils::File::conformName(QString::fromStdString(m_name)) + "/";
    QString path = pluginDir + "data/training.data";
    QFile file(path);

    if(file.open(QFile::WriteOnly | QFile::Text) == false)
        throw CException(CoreExCode::INVALID_FILE, "Unable to create file classes.txt", __func__, __FILE__, __LINE__);

    QTextStream stream(&file);
    stream << "classes = " << m_classCount << "\n";
    stream << "train = " << pluginDir + "data/train.txt\n";
    stream << "valid = " << pluginDir + "data/eval.txt\n";
    stream << "names = " << pluginDir + "data/classes.txt\n";
    stream << "backup = " << pluginDir + "data/models\n";
    stream << "metrics = " << pluginDir + "data/metrics.txt";
}

void CYoloTrain::createConfigFile()
{
    auto paramPtr = std::dynamic_pointer_cast<CYoloTrainParam>(m_pParam);
    paramPtr->m_epochs = m_classCount * 2000;
    int burnin = (int)(paramPtr->m_epochs * 0.05);
    int step1 = (int)(paramPtr->m_epochs * 0.8);
    int step2 = (int)(paramPtr->m_epochs * 0.9);
    int filters = (m_classCount + 5) * 3;

    QString pluginDir = QString::fromStdString(Utils::Plugin::getCppPath()) + "/" + Utils::File::conformName(QString::fromStdString(m_name)) + "/";
    QString templatePath = pluginDir + "data/config/" + _modelConfigFiles[paramPtr->m_model];
    QString configPath = pluginDir + "data/config/training.cfg";
    paramPtr->m_configPath = configPath.toStdString();

    QFile templateFile(templatePath);
    if(templateFile.open(QFile::ReadOnly | QFile::Text) == false)
        throw CException(CoreExCode::INVALID_PARAMETER, QObject::tr("Unable to read config template file.").toStdString(), __func__, __FILE__, __LINE__);

    QTextStream txtStream(&templateFile);
    auto templateContent = txtStream.readAll();
    templateFile.close();

    auto newContent = templateContent.replace("_batch_", QString::number(paramPtr->m_batchSize));
    newContent = newContent.replace("_subdivision_", QString::number(paramPtr->m_subdivision));
    newContent = newContent.replace("_width_", QString::number(paramPtr->m_inputWidth));
    newContent = newContent.replace("_height_", QString::number(paramPtr->m_inputHeight));
    newContent = newContent.replace("_momentum_", QString::number(paramPtr->m_momentum));
    newContent = newContent.replace("_decay_", QString::number(paramPtr->m_weightDecay));
    newContent = newContent.replace("_lr_", QString::number(paramPtr->m_learningRate));
    newContent = newContent.replace("_burnin_", QString::number(burnin));
    newContent = newContent.replace("_epochs_", QString::number(paramPtr->m_epochs));
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
    QFile configFile(QString::fromStdString(paramPtr->m_configPath));

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
        paramPtr->m_batchSize = match.captured(1).toInt();

    //Subdivision
    re.setPattern("subdivisions *= *([0-9]+)");
    match = re.match(config);

    if(match.hasMatch())
        paramPtr->m_subdivision = match.captured(1).toInt();

    //Input width
    re.setPattern("width *= *([0-9]+)");
    match = re.match(config);

    if(match.hasMatch())
        paramPtr->m_inputWidth = match.captured(1).toInt();

    //Input height
    re.setPattern("width *= *([0-9]+)");
    match = re.match(config);

    if(match.hasMatch())
        paramPtr->m_inputHeight = match.captured(1).toInt();

    //Momentum
    re.setPattern("momentum *= *([0-9.]+)");
    match = re.match(config);

    if(match.hasMatch())
        paramPtr->m_momentum = match.captured(1).toFloat();

    //Decay
    re.setPattern("decay *= *([0-9.]+)");
    match = re.match(config);

    if(match.hasMatch())
        paramPtr->m_weightDecay = match.captured(1).toFloat();

    //Learning rate
    re.setPattern("learning_rate *= *([0-9.]+)");
    match = re.match(config);

    if(match.hasMatch())
        paramPtr->m_learningRate = match.captured(1).toFloat();

    //Epochs
    re.setPattern("max_batches *= *([0-9]+)");
    match = re.match(config);

    if(match.hasMatch())
        paramPtr->m_epochs = match.captured(1).toInt();
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

void CYoloTrain::initParamsLogging()
{
    auto paramPtr = std::dynamic_pointer_cast<CYoloTrainParam>(m_pParam);
    if(paramPtr)
    {
        std::map<std::string, std::string> params;
        paramPtr->m_modelName = _modelNames[paramPtr->m_model];
        paramPtr->m_classes = m_classCount;
        params["Model"] = paramPtr->m_model;
        params["Batch size"] = std::to_string(paramPtr->m_batchSize);
        params["Epochs"] = std::to_string(paramPtr->m_epochs);
        params["Classes"] = std::to_string(paramPtr->m_classes);
        params["Input width"] = std::to_string(paramPtr->m_inputWidth);
        params["Input height"] = std::to_string(paramPtr->m_inputHeight);
        params["Learning rate"] = std::to_string(paramPtr->m_learningRate);
        params["Momentum"] = std::to_string(paramPtr->m_momentum);
        params["Weight decay"] = std::to_string(paramPtr->m_weightDecay);
        logParams(params);
    }
}

void CYoloTrain::launchTraining()
{
    auto paramPtr = std::dynamic_pointer_cast<CYoloTrainParam>(m_pParam);
    QString pluginDir = QString::fromStdString(Utils::Plugin::getCppPath()) + "/" + Utils::File::conformName(QString::fromStdString(m_name)) + "/";
    QString dataFilePath = pluginDir + "data/training.data";
    QString configFilePath = QString::fromStdString(paramPtr->m_configPath);
    QString weightsFilePath = pluginDir + "data/models/pretrained/" + _modelWeightFiles[paramPtr->m_model];
    QString metricsFilePath = pluginDir + "data/metrics.txt";
    QString logFilePath = pluginDir + "data/log.txt";
    QString darknetExe = pluginDir + "darknet";

    QStringList args;
    args << "detector" << "train" << dataFilePath << configFilePath << weightsFilePath << "-dont_show" << "-map" << "-log_metrics";

    QProcess proc;
    proc.setProcessChannelMode(QProcess::MergedChannels);
    proc.setStandardOutputFile(logFilePath, QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text);
    proc.start(darknetExe, args);
    proc.waitForStarted();

    QFile metricsFile(metricsFilePath);
    while(!metricsFile.exists(metricsFilePath));
    metricsFile.open(QFile::ReadOnly | QFile::Text);
    QTextStream metricsStream(&metricsFile);

    //MLflow is quiet slow, we log metrics asynchronously
    m_mlflowLogFreq = std::max(1, paramPtr->m_epochs / 100);
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

    //Wait for MLflow logging process
    emit m_signalHandler->doLog("Waiting for MLflow logging process...");
    mlflowFuture.wait();

    //Log config file
    logArtifact(configFilePath.toStdString());
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
