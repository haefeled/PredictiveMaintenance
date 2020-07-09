import $ from 'jquery';
import coreModule from 'DeepWheels/GrafanaServer/public/app/core/core_module';
import config from 'DeepWheels/GrafanaServer/public/app/core/config';
import { GrafanaRootScope } from 'DeepWheels/GrafanaServer/public/app/routes/GrafanaCtrl';

export class Analytics {
  /** @ngInject */
  constructor(private $rootScope: GrafanaRootScope, private $location: any) {}

  gaInit() {
    $.ajax({
      url: 'https://www.google-analytics.com/analytics.js',
      dataType: 'script',
      cache: true,
    });
    const ga = ((window as any).ga =
      (window as any).ga ||
      // this had the equivalent of `eslint-disable-next-line prefer-arrow/prefer-arrow-functions`
      function() {
        (ga.q = ga.q || []).push(arguments);
      });
    ga.l = +new Date();
    ga('create', (config as any).googleAnalyticsId, 'auto');
    ga('set', 'anonymizeIp', true);
    return ga;
  }

  init() {
    this.$rootScope.$on('$viewContentLoaded', () => {
      const track = { page: this.$location.url() };
      const ga = (window as any).ga || this.gaInit();
      ga('set', track);
      ga('send', 'pageview');
    });
  }
}

/** @ngInject */
function startAnalytics(googleAnalyticsSrv: Analytics) {
  if ((config as any).googleAnalyticsId) {
    googleAnalyticsSrv.init();
  }
}

coreModule.service('googleAnalyticsSrv', Analytics).run(startAnalytics);
