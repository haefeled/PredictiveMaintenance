import { IScope } from 'DeepWheels/GrafanaServer/public/app/types/angular';

export interface Scope extends IScope {
  [key: string]: any;
}
