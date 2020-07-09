const graphitePlugin = async () =>
  await import(/* webpackChunkName: "graphitePlugin" */ 'DeepWheels/GrafanaServer/public/app/plugins/datasource/graphite/module');
const cloudwatchPlugin = async () =>
  await import(/* webpackChunkName: "cloudwatchPlugin" */ 'DeepWheels/GrafanaServer/public/app/plugins/datasource/cloudwatch/module');
const dashboardDSPlugin = async () =>
  await import(/* webpackChunkName "dashboardDSPlugin" */ 'DeepWheels/GrafanaServer/public/app/plugins/datasource/dashboard/module');
const elasticsearchPlugin = async () =>
  await import(/* webpackChunkName: "elasticsearchPlugin" */ 'DeepWheels/GrafanaServer/public/app/plugins/datasource/elasticsearch/module');
const opentsdbPlugin = async () =>
  await import(/* webpackChunkName: "opentsdbPlugin" */ 'DeepWheels/GrafanaServer/public/app/plugins/datasource/opentsdb/module');
const grafanaPlugin = async () =>
  await import(/* webpackChunkName: "grafanaPlugin" */ 'DeepWheels/GrafanaServer/public/app/plugins/datasource/grafana/module');
const influxdbPlugin = async () =>
  await import(/* webpackChunkName: "influxdbPlugin" */ 'DeepWheels/GrafanaServer/public/app/plugins/datasource/influxdb/module');
const lokiPlugin = async () => await import(/* webpackChunkName: "lokiPlugin" */ 'DeepWheels/GrafanaServer/public/app/plugins/datasource/loki/module');
const jaegerPlugin = async () =>
  await import(/* webpackChunkName: "jaegerPlugin" */ 'DeepWheels/GrafanaServer/public/app/plugins/datasource/jaeger/module');
const zipkinPlugin = async () =>
  await import(/* webpackChunkName: "zipkinPlugin" */ 'DeepWheels/GrafanaServer/public/app/plugins/datasource/zipkin/module');
const mixedPlugin = async () =>
  await import(/* webpackChunkName: "mixedPlugin" */ 'DeepWheels/GrafanaServer/public/app/plugins/datasource/mixed/module');
const mysqlPlugin = async () =>
  await import(/* webpackChunkName: "mysqlPlugin" */ 'DeepWheels/GrafanaServer/public/app/plugins/datasource/mysql/module');
const postgresPlugin = async () =>
  await import(/* webpackChunkName: "postgresPlugin" */ 'DeepWheels/GrafanaServer/public/app/plugins/datasource/postgres/module');
const prometheusPlugin = async () =>
  await import(/* webpackChunkName: "prometheusPlugin" */ 'DeepWheels/GrafanaServer/public/app/plugins/datasource/prometheus/module');
const mssqlPlugin = async () =>
  await import(/* webpackChunkName: "mssqlPlugin" */ 'DeepWheels/GrafanaServer/public/app/plugins/datasource/mssql/module');
const testDataDSPlugin = async () =>
  await import(/* webpackChunkName: "testDataDSPlugin" */ 'DeepWheels/GrafanaServer/public/app/plugins/datasource/testdata/module');
const stackdriverPlugin = async () =>
  await import(/* webpackChunkName: "stackdriverPlugin" */ 'DeepWheels/GrafanaServer/public/app/plugins/datasource/stackdriver/module');
const azureMonitorPlugin = async () =>
  await import(
    /* webpackChunkName: "azureMonitorPlugin" */ 'DeepWheels/GrafanaServer/public/app/plugins/datasource/grafana-azure-monitor-datasource/module'
  );

import * as textPanel from 'DeepWheels/GrafanaServer/public/app/plugins/panel/text/module';
import * as text2Panel from 'DeepWheels/GrafanaServer/public/app/plugins/panel/text2/module';
import * as graph2Panel from 'DeepWheels/GrafanaServer/public/app/plugins/panel/graph2/module';
import * as graphPanel from 'DeepWheels/GrafanaServer/public/app/plugins/panel/graph/module';
import * as dashListPanel from 'DeepWheels/GrafanaServer/public/app/plugins/panel/dashlist/module';
import * as pluginsListPanel from 'DeepWheels/GrafanaServer/public/app/plugins/panel/pluginlist/module';
import * as alertListPanel from 'DeepWheels/GrafanaServer/public/app/plugins/panel/alertlist/module';
import * as annoListPanel from 'DeepWheels/GrafanaServer/public/app/plugins/panel/annolist/module';
import * as heatmapPanel from 'DeepWheels/GrafanaServer/public/app/plugins/panel/heatmap/module';
import * as tablePanel from 'DeepWheels/GrafanaServer/public/app/plugins/panel/table/module';
import * as oldTablePanel from 'DeepWheels/GrafanaServer/public/app/plugins/panel/table-old/module';
import * as singlestatPanel from 'DeepWheels/GrafanaServer/public/app/plugins/panel/singlestat/module';
import * as singlestatPanel2 from 'DeepWheels/GrafanaServer/public/app/plugins/panel/stat/module';
import * as gettingStartedPanel from 'DeepWheels/GrafanaServer/public/app/plugins/panel/gettingstarted/module';
import * as gaugePanel from 'DeepWheels/GrafanaServer/public/app/plugins/panel/gauge/module';
import * as pieChartPanel from 'DeepWheels/GrafanaServer/public/app/plugins/panel/piechart/module';
import * as barGaugePanel from 'DeepWheels/GrafanaServer/public/app/plugins/panel/bargauge/module';
import * as logsPanel from 'DeepWheels/GrafanaServer/public/app/plugins/panel/logs/module';
import * as newsPanel from 'DeepWheels/GrafanaServer/public/app/plugins/panel/news/module';
import * as homeLinksPanel from 'DeepWheels/GrafanaServer/public/app/plugins/panel/homelinks/module';
import * as welcomeBanner from 'DeepWheels/GrafanaServer/public/app/plugins/panel/welcome/module';

const builtInPlugins: any = {
  'app/plugins/datasource/graphite/module': graphitePlugin,
  'app/plugins/datasource/cloudwatch/module': cloudwatchPlugin,
  'app/plugins/datasource/dashboard/module': dashboardDSPlugin,
  'app/plugins/datasource/elasticsearch/module': elasticsearchPlugin,
  'app/plugins/datasource/opentsdb/module': opentsdbPlugin,
  'app/plugins/datasource/grafana/module': grafanaPlugin,
  'app/plugins/datasource/influxdb/module': influxdbPlugin,
  'app/plugins/datasource/loki/module': lokiPlugin,
  'app/plugins/datasource/jaeger/module': jaegerPlugin,
  'app/plugins/datasource/zipkin/module': zipkinPlugin,
  'app/plugins/datasource/mixed/module': mixedPlugin,
  'app/plugins/datasource/mysql/module': mysqlPlugin,
  'app/plugins/datasource/postgres/module': postgresPlugin,
  'app/plugins/datasource/mssql/module': mssqlPlugin,
  'app/plugins/datasource/prometheus/module': prometheusPlugin,
  'app/plugins/datasource/testdata/module': testDataDSPlugin,
  'app/plugins/datasource/stackdriver/module': stackdriverPlugin,
  'app/plugins/datasource/grafana-azure-monitor-datasource/module': azureMonitorPlugin,

  'app/plugins/panel/text/module': textPanel,
  'app/plugins/panel/text2/module': text2Panel,
  'app/plugins/panel/graph2/module': graph2Panel,
  'app/plugins/panel/graph/module': graphPanel,
  'app/plugins/panel/dashlist/module': dashListPanel,
  'app/plugins/panel/pluginlist/module': pluginsListPanel,
  'app/plugins/panel/alertlist/module': alertListPanel,
  'app/plugins/panel/annolist/module': annoListPanel,
  'app/plugins/panel/heatmap/module': heatmapPanel,
  'app/plugins/panel/table/module': tablePanel,
  'app/plugins/panel/table-old/module': oldTablePanel,
  'app/plugins/panel/news/module': newsPanel,
  'app/plugins/panel/singlestat/module': singlestatPanel,
  'app/plugins/panel/stat/module': singlestatPanel2,
  'app/plugins/panel/gettingstarted/module': gettingStartedPanel,
  'app/plugins/panel/gauge/module': gaugePanel,
  'app/plugins/panel/piechart/module': pieChartPanel,
  'app/plugins/panel/bargauge/module': barGaugePanel,
  'app/plugins/panel/logs/module': logsPanel,
  'app/plugins/panel/homelinks/module': homeLinksPanel,
  'app/plugins/panel/welcome/module': welcomeBanner,
};

export default builtInPlugins;
