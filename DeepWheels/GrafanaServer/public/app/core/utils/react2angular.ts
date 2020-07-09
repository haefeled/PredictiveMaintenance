import coreModule from 'DeepWheels/GrafanaServer/public/app/core/core_module';
import { provideTheme } from 'DeepWheels/GrafanaServer/public/app/core/utils/ConfigProvider';

export function react2AngularDirective(name: string, component: any, options: any) {
  coreModule.directive(name, [
    'reactDirective',
    reactDirective => {
      return reactDirective(provideTheme(component), options);
    },
  ]);
}
