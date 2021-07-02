#ifndef YOLOTRAIN_HPP
#define YOLOTRAIN_HPP

#include "CPluginProcessInterface.hpp"
#include "YoloTrainGlobal.hpp"
#include "YoloTrainProcess.h"
#include "YoloTrainWidget.h"

//-----------------------------------//
//----- Global plugin interface -----//
//-----------------------------------//
class YOLOTRAIN_EXPORT CYoloTrainInterface : public QObject, public CPluginProcessInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "ikomia.plugin.process")
    Q_INTERFACES(CPluginProcessInterface)

    public:

        virtual std::shared_ptr<CTaskFactory> getProcessFactory()
        {
            return std::make_shared<CYoloTrainFactory>();
        }

        virtual std::shared_ptr<CWidgetFactory> getWidgetFactory()
        {
            return std::make_shared<CYoloTrainWidgetFactory>();
        }
};

#endif // YOLOTRAIN_HPP
