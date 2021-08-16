#include "YoloTrainWidget.h"

CYoloTrainWidget::CYoloTrainWidget(QWidget *parent): CWorkflowTaskWidget(parent)
{
    m_pParam = std::make_shared<CYoloTrainParam>();
    init();
}

CYoloTrainWidget::CYoloTrainWidget(WorkflowTaskParamPtr pParam, QWidget *parent): CWorkflowTaskWidget(parent)
{
    m_pParam = std::dynamic_pointer_cast<CYoloTrainParam>(pParam);
    if(m_pParam == nullptr)
        m_pParam = std::make_shared<CYoloTrainParam>();

    init();
}

void CYoloTrainWidget::init()
{
    assert(m_pParam);

    m_pComboModel = addCombo(tr("Model"));
    m_pComboModel->addItem("YOLOv4", CYoloTrainParam::YOLOV4);
    m_pComboModel->addItem("YOLOv3", CYoloTrainParam::YOLOV3);
    m_pComboModel->addItem("Tiny YOLOv4", CYoloTrainParam::TINY_YOLOV4);
    m_pComboModel->addItem("Tiny YOLOv3", CYoloTrainParam::TINY_YOLOV3);
    m_pComboModel->addItem("EfficientNet B0 YOLOv3", CYoloTrainParam::ENET_B0_YOLOV3);
    m_pComboModel->setCurrentIndex(m_pComboModel->findData(std::stoi(m_pParam->m_cfg["model"])));

    m_pSpinWidth = addSpin("Input width", std::stoi(m_pParam->m_cfg["inputWidth"]), 1, 1024, 1);
    m_pSpinHeight = addSpin("Input height", std::stoi(m_pParam->m_cfg["inputHeight"]), 1, 1024, 1);
    m_pSpinTrainEvalRatio = addDoubleSpin("Train/Eval split ratio", std::stod(m_pParam->m_cfg["splitRatio"]), 0.1, 0.9, 0.1, 1);
    m_pSpinBatchSize = addSpin("Batch size", std::stoi(m_pParam->m_cfg["batchSize"]), 1, 64, 1);
    m_pSpinLr = addDoubleSpin("Learning rate", std::stod(m_pParam->m_cfg["learningRate"]), 0.0001, 0.1, 0.001, 4);
    m_pSpinMomentum =  addDoubleSpin("Momentum", std::stod(m_pParam->m_cfg["momentum"]), 0.0, 1.0, 0.01, 2);
    m_pSpinDecay = addDoubleSpin("Weight decay", std::stod(m_pParam->m_cfg["weightDecay"]), 0.0, 1.0, 0.0001, 4);
    m_pSpinSubdivision = addSpin("Subdivision", std::stoi(m_pParam->m_cfg["subdivision"]), 4, 64, 2);
    m_pCheckAutoConfig = addCheck("Auto configuration", std::stoi(m_pParam->m_cfg["autoConfig"]));
    m_pBrowseFile = addBrowseFile("Configuration file path", QString::fromStdString(m_pParam->m_cfg["configPath"]), "Select configuration file");
    m_pBrowseFile->setEnabled(std::stoi(m_pParam->m_cfg["autoConfig"]) == false);
    m_pBrowseOutFolder = addBrowseFolder("Output folder", QString::fromStdString(m_pParam->m_cfg["outputPath"]), "Select output folder");

    connect(m_pCheckAutoConfig, &QCheckBox::stateChanged, [&](int state)
    {
        m_pBrowseFile->setEnabled(state == false);
    });
}

void CYoloTrainWidget::onApply()
{
    m_pParam->m_cfg["model"] = m_pComboModel->currentData().toString().toStdString();
    m_pParam->m_cfg["subdivision"] = std::to_string(m_pSpinSubdivision->value());
    m_pParam->m_cfg["inputWidth"] = std::to_string(m_pSpinWidth->value());
    m_pParam->m_cfg["inputHeight"] = std::to_string(m_pSpinHeight->value());
    m_pParam->m_cfg["splitRatio"] = std::to_string(m_pSpinTrainEvalRatio->value());
    m_pParam->m_cfg["batchSize"] = std::to_string(m_pSpinBatchSize->value());
    m_pParam->m_cfg["learningRate"] = std::to_string(m_pSpinLr->value());
    m_pParam->m_cfg["momentum"] = std::to_string(m_pSpinMomentum->value());
    m_pParam->m_cfg["weightDecay"] = std::to_string(m_pSpinDecay->value());
    m_pParam->m_cfg["autoConfig"] = std::to_string(m_pCheckAutoConfig->isChecked());
    m_pParam->m_cfg["configPath"] = m_pBrowseFile->getPath().toStdString();
    m_pParam->m_cfg["outputPath"] = m_pBrowseOutFolder->getPath().toStdString();
    emit doApplyProcess(m_pParam);
}
