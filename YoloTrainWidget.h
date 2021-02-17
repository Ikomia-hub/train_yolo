#ifndef CYOLOTRAINWIDGET_H
#define CYOLOTRAINWIDGET_H

#include "YoloTrainGlobal.hpp"
#include "YoloTrainProcess.h"
#include "Core/CWidgetFactory.hpp"

//----------------------------//
//----- CYoloTrainWidget -----//
//----------------------------//
class YOLOTRAIN_EXPORT CYoloTrainWidget: public CProtocolTaskWidget
{
    public:

        CYoloTrainWidget(QWidget *parent = Q_NULLPTR);
        CYoloTrainWidget(ProtocolTaskParamPtr pParam, QWidget *parent = Q_NULLPTR);

    private:

        void    init() override;

    private:

        std::shared_ptr<CYoloTrainParam>   m_pParam = nullptr;
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

        virtual ProtocolTaskWidgetPtr   create(ProtocolTaskParamPtr pParam)
        {
            return std::make_shared<CYoloTrainWidget>(pParam);
        }
};


#endif // CYOLOTRAINWIDGET_H
