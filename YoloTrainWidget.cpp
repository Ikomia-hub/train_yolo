#include "YoloTrainWidget.h"

CYoloTrainWidget::CYoloTrainWidget(QWidget *parent): CProtocolTaskWidget(parent)
{
    m_pParam = std::make_shared<CYoloTrainParam>();
    init();
}

CYoloTrainWidget::CYoloTrainWidget(ProtocolTaskParamPtr pParam, QWidget *parent): CProtocolTaskWidget(parent)
{
    m_pParam = std::dynamic_pointer_cast<CYoloTrainParam>(pParam);
    if(m_pParam == nullptr)
        m_pParam = std::make_shared<CYoloTrainParam>();

    init();
}

void CYoloTrainWidget::init()
{
    assert(m_pParam);

    auto pComboModel = addCombo(tr("Model"));
    pComboModel->addItem("YOLOv4", CYoloTrainParam::YOLOV4);
    pComboModel->addItem("YOLOv3", CYoloTrainParam::YOLOV3);
    pComboModel->addItem("Tiny YOLOv4", CYoloTrainParam::TINY_YOLOV4);
    pComboModel->addItem("Tiny YOLOv3", CYoloTrainParam::TINY_YOLOV3);
    pComboModel->addItem("EfficientNet B0 YOLOv3", CYoloTrainParam::ENET_B0_YOLOV3);
    pComboModel->setCurrentIndex(pComboModel->findData(std::stoi(m_pParam->m_cfg["model"])));

    auto pSpinWidth = addSpin("Input width", std::stoi(m_pParam->m_cfg["inputWidth"]), 1, 1024, 1);
    auto pSpinHeight = addSpin("Input height", std::stoi(m_pParam->m_cfg["inputHeight"]), 1, 1024, 1);
    auto pSpinTrainEvalRatio = addDoubleSpin("Train/Eval split ratio", std::stod(m_pParam->m_cfg["splitRatio"]), 0.1, 0.9, 0.1, 1);
    auto pSpinBatchSize = addSpin("Batch size", std::stoi(m_pParam->m_cfg["batchSize"]), 1, 64, 1);
    auto pSpinLr = addDoubleSpin("Learning rate", std::stod(m_pParam->m_cfg["learningRate"]), 0.0001, 0.1, 0.001, 4);
    auto pSpinMomentum =  addDoubleSpin("Momentum", std::stod(m_pParam->m_cfg["momentum"]), 0.0, 1.0, 0.01, 2);
    auto pSpinDecay = addDoubleSpin("Weight decay", std::stod(m_pParam->m_cfg["weightDecay"]), 0.0, 1.0, 0.0001, 4);
    auto pSpinSubdivision = addSpin("Subdivision", std::stoi(m_pParam->m_cfg["subdivision"]), 4, 64, 2);
    auto pCheckAutoConfig = addCheck("Auto configuration", std::stoi(m_pParam->m_cfg["autoConfig"]));
    auto pBrowseFile = addBrowseFile("Configuration file path", QString::fromStdString(m_pParam->m_cfg["configPath"]), "Select configuration file");
    pBrowseFile->setEnabled(std::stoi(m_pParam->m_cfg["autoConfig"]) == false);
    auto pBrowseOutFolder = addBrowseFolder("Output folder", QString::fromStdString(m_pParam->m_cfg["outputPath"]), "Select output folder");

    connect(pCheckAutoConfig, &QCheckBox::stateChanged, [=](int state)
    {
        pBrowseFile->setEnabled(state == false);
    });

    connect(m_pApplyBtn, &QPushButton::clicked, [=]
    {
        m_pParam->m_cfg["model"] = pComboModel->currentData().toString().toStdString();
        m_pParam->m_cfg["subdivision"] = std::to_string(pSpinSubdivision->value());
        m_pParam->m_cfg["inputWidth"] = std::to_string(pSpinWidth->value());
        m_pParam->m_cfg["inputHeight"] = std::to_string(pSpinHeight->value());
        m_pParam->m_cfg["splitRatio"] = std::to_string(pSpinTrainEvalRatio->value());
        m_pParam->m_cfg["batchSize"] = std::to_string(pSpinBatchSize->value());
        m_pParam->m_cfg["learningRate"] = std::to_string(pSpinLr->value());
        m_pParam->m_cfg["momentum"] = std::to_string(pSpinMomentum->value());
        m_pParam->m_cfg["weightDecay"] = std::to_string(pSpinDecay->value());
        m_pParam->m_cfg["autoConfig"] = std::to_string(pCheckAutoConfig->isChecked());
        m_pParam->m_cfg["configPath"] = pBrowseFile->getPath().toStdString();
        m_pParam->m_cfg["outputPath"] = pBrowseOutFolder->getPath().toStdString();
        emit doApplyProcess(m_pParam);
    });
}
