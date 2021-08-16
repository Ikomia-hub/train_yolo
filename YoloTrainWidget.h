#ifndef CYOLOTRAINWIDGET_H
#define CYOLOTRAINWIDGET_H

#include "YoloTrainGlobal.hpp"
#include "YoloTrainProcess.h"
#include "Core/CWidgetFactory.hpp"

//----------------------------//
//----- CYoloTrainWidget -----//
//----------------------------//
class YOLOTRAIN_EXPORT CYoloTrainWidget: public CWorkflowTaskWidget
{
    public:

        CYoloTrainWidget(QWidget *parent = Q_NULLPTR);
        CYoloTrainWidget(WorkflowTaskParamPtr pParam, QWidget *parent = Q_NULLPTR);

        void    onApply() override;

    private:

        void    init();

    private:

        std::shared_ptr<CYoloTrainParam>   m_pParam = nullptr;
        QDoubleSpinBox*     m_pSpinTrainEvalRatio = nullptr;
        QDoubleSpinBox*     m_pSpinLr = nullptr;
        QDoubleSpinBox*     m_pSpinMomentum = nullptr;
        QDoubleSpinBox*     m_pSpinDecay = nullptr;
        QSpinBox*           m_pSpinWidth = nullptr;
        QSpinBox*           m_pSpinHeight = nullptr;
        QSpinBox*           m_pSpinBatchSize = nullptr;
        QSpinBox*           m_pSpinSubdivision = nullptr;
        QComboBox*          m_pComboModel = nullptr;
        QCheckBox*          m_pCheckAutoConfig = nullptr;
        CBrowseFileWidget*  m_pBrowseFile = nullptr;
        CBrowseFileWidget*  m_pBrowseOutFolder = nullptr;
};

//-----------------------------------//
//----- CYoloTrainWidgetFactory -----//
//-----------------------------------//
class YOLOTRAIN_EXPORT CYoloTrainWidgetFactory : public CWidgetFactory
{
    public:

        CYoloTrainWidgetFactory()
        {
            m_name = QObject::tr("YoloTrain").toStdString();
        }

        virtual WorkflowTaskWidgetPtr   create(WorkflowTaskParamPtr pParam)
        {
            return std::make_shared<CYoloTrainWidget>(pParam);
        }
};


#endif // CYOLOTRAINWIDGET_H
