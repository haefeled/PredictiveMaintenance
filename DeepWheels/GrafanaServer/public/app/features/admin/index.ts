import AdminEditUserCtrl from './AdminEditUserCtrl';
import AdminEditOrgCtrl from './AdminEditOrgCtrl';

import coreModule from 'DeepWheels/GrafanaServer/public/app/core/core_module';
import { NavModelSrv } from 'DeepWheels/GrafanaServer/public/app/core/core';

class AdminHomeCtrl {
  navModel: any;

  /** @ngInject */
  constructor(navModelSrv: NavModelSrv) {
    this.navModel = navModelSrv.getNav('admin');
  }
}

coreModule.controller('AdminEditUserCtrl', AdminEditUserCtrl);
coreModule.controller('AdminEditOrgCtrl', AdminEditOrgCtrl);
coreModule.controller('AdminHomeCtrl', AdminHomeCtrl);
