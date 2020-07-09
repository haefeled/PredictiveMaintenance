import { SortOrder } from 'DeepWheels/GrafanaServer/public/app/core/utils/explore';

export interface Options {
  showLabels: boolean;
  showTime: boolean;
  wrapLogMessage: boolean;
  sortOrder: SortOrder;
}
