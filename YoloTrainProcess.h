#ifndef YOLOTRAIN_H
#define YOLOTRAIN_H

#include <QTextStream>
#include <QFile>
#include "YoloTrainGlobal.hpp"
#include "Core/CProcessFactory.hpp"
#include "Core/CMlflowTrainProcess.h"

//---------------------------//
//----- CYoloTrainParam -----//
//---------------------------//
class YOLOTRAIN_EXPORT CYoloTrainParam: public CDnnTrainProcessParam
{
    public:

        enum Model {YOLOV4, YOLOV3, TINY_YOLOV4, TINY_YOLOV3, ENET_B0_YOLOV3};

        CYoloTrainParam() : CDnnTrainProcessParam()
        {
            m_batchSize = 32;
        }

        void        setParamMap(const UMapString& paramMap) override
        {
            CDnnTrainProcessParam::setParamMap(paramMap);
            m_model = std::stoi(paramMap.at("model"));
            m_gpuCount = std::stoi(paramMap.at("gpuCount"));
            m_subdivision = std::stoi(paramMap.at("subdivision"));
            m_inputWidth = std::stoi(paramMap.at("inputWidth"));
            m_inputHeight = std::stoi(paramMap.at("inputHeight"));
            m_splitRatio = std::stof(paramMap.at("splitRatio"));
            m_bAutoConfig = std::stoi(paramMap.at("bAutoConfig"));
            m_configPath = paramMap.at("configPath");
        }

        UMapString  getParamMap() const override
        {
            auto paramMap = CDnnTrainProcessParam::getParamMap();
            paramMap.insert(std::make_pair("model", std::to_string(m_model)));
            paramMap.insert(std::make_pair("gpuCount", std::to_string(m_gpuCount)));
            paramMap.insert(std::make_pair("subdivision", std::to_string(m_subdivision)));
            paramMap.insert(std::make_pair("inputWidth", std::to_string(m_inputWidth)));
            paramMap.insert(std::make_pair("inputHeight", std::to_string(m_inputHeight)));
            paramMap.insert(std::make_pair("splitRatio", std::to_string(m_splitRatio)));
            paramMap.insert(std::make_pair("bAutoConfig", std::to_string(m_bAutoConfig)));
            paramMap.insert(std::make_pair("configPath", m_configPath));
            return paramMap;
        }

    public:

        int         m_model = TINY_YOLOV3;
        int         m_gpuCount = 1;
        int         m_subdivision = 16;
        int         m_inputWidth = 416;
        int         m_inputHeight = 416;
        float       m_splitRatio = 0.9;
        bool        m_bAutoConfig = true;
        std::string m_configPath = "";
};

//----------------------//
//----- CYoloTrain -----//
//----------------------//
class YOLOTRAIN_EXPORT CYoloTrain: public CMlflowTrainProcess
{
    public:

        CYoloTrain();
        CYoloTrain(const std::string& name, const std::shared_ptr<CYoloTrainParam>& paramPtr);

        size_t      getProgressSteps() override;
        void        run() override;
        void        stop() override;

    private:

        using YoloMetrics = std::map<std::string, float>;

        void        prepareData();

        void        createAnnotationFiles(const QJsonDocument& json) const;
        void        createClassNamesFile(const QJsonDocument& json);
        void        createGlobalDataFile();
        void        createConfigFile();

        void        updateParamFromConfigFile();

        void        splitTrainEval(const QJsonDocument& json, float ratio = 0.9);

        void        initParamsLogging();

        void        launchTraining();

        void        loadMetrics(QTextStream &stream);

        void        deleteTrainingFiles();

    private:

        int                     m_classCount = 0;
        int                     m_mlflowLogFreq = 1;
        std::atomic_bool        m_bStop{false};
        std::atomic_bool        m_bFinished{false};
        QFile                   m_logFile;
        std::queue<YoloMetrics> m_metricsQueue;
};

//-----------------------------//
//----- CYoloTrainFactory -----//
//-----------------------------//
class YOLOTRAIN_EXPORT CYoloTrainFactory : public CProcessFactory
{
    public:

        CYoloTrainFactory()
        {
            m_info.m_name = QObject::tr("YoloTrain").toStdString();
            m_info.m_shortDescription = QObject::tr("Train YOLO neural network with darknet framework").toStdString();
            m_info.m_description = QObject::tr("Train YOLO neural network with darknet framework.").toStdString();
            m_info.m_path = QObject::tr("Plugins/C++/Train").toStdString();
            m_info.m_version = "1.0.0";
            m_info.m_iconPath = "Icon/icon.png";
            m_info.m_authors = "Ikomia team";
            m_info.m_article = "";
            m_info.m_license = "MIT License";
            m_info.m_repo = "https://github.com/Ikomia-dev/IkomiaPluginsCpp";
            m_info.m_keywords = "deep,learning,detection,yolo,darknet";
        }

        virtual ProtocolTaskPtr create(const ProtocolTaskParamPtr& pParam) override
        {
            auto paramPtr = std::dynamic_pointer_cast<CYoloTrainParam>(pParam);
            if(paramPtr != nullptr)
                return std::make_shared<CYoloTrain>(m_info.m_name, paramPtr);
            else
                return create();
        }
        virtual ProtocolTaskPtr create() override
        {
            auto paramPtr = std::make_shared<CYoloTrainParam>();
            assert(paramPtr != nullptr);
            return std::make_shared<CYoloTrain>(m_info.m_name, paramPtr);
        }
};

#endif // YOLOTRAIN_H
