#ifndef YOLOTRAIN_H
#define YOLOTRAIN_H

#include <QTextStream>
#include <QFile>
#include "YoloTrainGlobal.hpp"
#include "Core/CTaskFactory.hpp"
#include "Core/CMlflowTrainTask.h"
#include "Main/CoreTools.hpp"

//---------------------------//
//----- CYoloTrainParam -----//
//---------------------------//
class YOLOTRAIN_EXPORT CYoloTrainParam: public CWorkflowTaskParam
{
    public:

        CYoloTrainParam();
};

//----------------------//
//----- CYoloTrain -----//
//----------------------//
class YOLOTRAIN_EXPORT CYoloTrain: public CMlflowTrainTask
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

        void        launchTraining();

        void        loadMetrics(QTextStream &stream);

        void        deleteTrainingFiles();

    private:

        int                         m_classCount = 0;
        int                         m_mlflowLogFreq = 1;
        std::atomic_bool            m_bStop{false};
        std::atomic_bool            m_bFinished{false};
        QString                     m_outputFolder;
        QFile                       m_logFile;
        std::queue<YoloMetrics>     m_metricsQueue;
        const std::set<std::string> m_modelNames = {"yolov4", "yolov3", "tiny_yolov4", "tiny_yolov3", "enet_bo_yolov3"};
};

//-----------------------------//
//----- CYoloTrainFactory -----//
//-----------------------------//
class YOLOTRAIN_EXPORT CYoloTrainFactory : public CTaskFactory
{
    public:

        CYoloTrainFactory()
        {
            m_info.m_name = "train_yolo";
            m_info.m_shortDescription = QObject::tr("Train YOLO neural network with darknet framework").toStdString();
            m_info.m_description = QObject::tr("Train YOLO neural network with darknet framework.").toStdString();
            m_info.m_path = QObject::tr("Plugins/C++/Train").toStdString();
            m_info.m_version = "1.3.0";
            m_info.m_iconPath = "Icon/icon.png";
            m_info.m_authors = "Ikomia team";
            m_info.m_article = "";
            m_info.m_license = "MIT License";
            m_info.m_repo = "https://github.com/Ikomia-dev/IkomiaPluginsCpp";
            m_info.m_keywords = "deep,learning,detection,yolo,darknet," + Utils::Plugin::getArchitectureKeywords();
        }

        virtual WorkflowTaskPtr create(const WorkflowTaskParamPtr& pParam) override
        {
            auto paramPtr = std::dynamic_pointer_cast<CYoloTrainParam>(pParam);
            if(paramPtr != nullptr)
                return std::make_shared<CYoloTrain>(m_info.m_name, paramPtr);
            else
                return create();
        }
        virtual WorkflowTaskPtr create() override
        {
            auto paramPtr = std::make_shared<CYoloTrainParam>();
            assert(paramPtr != nullptr);
            return std::make_shared<CYoloTrain>(m_info.m_name, paramPtr);
        }
};

#endif // YOLOTRAIN_H
