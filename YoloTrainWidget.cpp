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
    pComboModel->setCurrentIndex(pComboModel->findData(m_pParam->m_model));

    auto pSpinWidth = addSpin("Input width", m_pParam->m_inputWidth, 1, 1024, 1);
    auto pSpinHeight = addSpin("Input height", m_pParam->m_inputHeight, 1, 1024, 1);
    auto pSpinTrainEvalRatio = addDoubleSpin("Train/Eval split ratio", m_pParam->m_splitRatio, 0.1, 0.9, 0.1, 1);
    auto pSpinBatchSize = addSpin("Batch size", m_pParam->m_batchSize, 1, 64, 1);
    auto pSpinLr = addDoubleSpin("Learning rate", m_pParam->m_learningRate, 0.0001, 0.1, 0.001, 4);
    auto pSpinMomentum =  addDoubleSpin("Momentum", m_pParam->m_momentum, 0.0, 1.0, 0.01, 2);
    auto pSpinDecay = addDoubleSpin("Weight decay", m_pParam->m_weightDecay, 0.0, 1.0, 0.0001, 4);
    auto pSpinSubdivision = addSpin("Subdivision", m_pParam->m_subdivision, 4, 64, 2);
    auto pCheckAutoConfig = addCheck("Auto configuration", m_pParam->m_bAutoConfig);
    auto pBrowseFile = addBrowseFile("Configuration file path", QString::fromStdString(m_pParam->m_configPath), "Select configuration file");
    pBrowseFile->setEnabled(m_pParam->m_bAutoConfig == false);

    connect(pCheckAutoConfig, &QCheckBox::stateChanged, [=](int state)
    {
        pBrowseFile->setEnabled(state == false);
    });

    connect(m_pApplyBtn, &QPushButton::clicked, [=]
    {
        m_pParam->m_model = pComboModel->currentData().toInt();
        m_pParam->m_subdivision = pSpinSubdivision->value();
        m_pParam->m_inputWidth = pSpinWidth->value();
        m_pParam->m_inputHeight = pSpinHeight->value();
        m_pParam->m_splitRatio = pSpinTrainEvalRatio->value();
        m_pParam->m_batchSize = pSpinBatchSize->value();
        m_pParam->m_learningRate = pSpinLr->value();
        m_pParam->m_momentum = pSpinMomentum->value();
        m_pParam->m_weightDecay = pSpinDecay->value();
        m_pParam->m_bAutoConfig = pCheckAutoConfig->isChecked();
        m_pParam->m_configPath = pBrowseFile->getPath().toStdString();
        emit doApplyProcess(m_pParam);
    });
}
