(window.webpackJsonp=window.webpackJsonp||[]).push([[16],{"/2Mn":function(e,t,n){"use strict";var r=n("q1tI"),a=n.n(r),o=n("Csm0"),i=n("ZFWI");t.a=function(e){var t=e.isReadOnly,n=e.onDelete,r=e.onSubmit,c=e.onTest;return a.a.createElement("div",{className:"gf-form-button-row"},!t&&a.a.createElement("button",{type:"submit",className:"btn btn-primary",disabled:t,onClick:function(e){return r(e)},"aria-label":o.selectors.pages.DataSource.saveAndTest},"Save & Test"),t&&a.a.createElement("button",{type:"submit",className:"btn btn-success",onClick:c},"Test"),a.a.createElement("button",{type:"button",className:"btn btn-danger",disabled:t,onClick:n,"aria-label":o.selectors.pages.DataSource.delete},"Delete"),a.a.createElement("a",{className:"btn btn-inverse",href:"".concat(i.b.appSubUrl,"/datasources")},"Back"))}},"5BCB":function(e,t,n){"use strict";var r=n("ZFWI"),a=n("NXk7"),o=n("WnbS"),i=n("3SGO"),c=n("HUMP"),u=n("okuo"),s=n("Vw/f"),l=n("FFN/"),d=n("Obii");function f(e){var t=[{id:"tsdb",title:"Time series databases",plugins:[]},{id:"logging",title:"Logging & document databases",plugins:[]},{id:"tracing",title:"Distributed tracing",plugins:[]},{id:"sql",title:"SQL",plugins:[]},{id:"cloud",title:"Cloud",plugins:[]},{id:"enterprise",title:"Enterprise plugins",plugins:[]},{id:"other",title:"Others",plugins:[]}].filter((function(e){return e})),n={},r={},a=[g({id:"grafana-splunk-datasource",name:"Splunk",description:"Visualize & explore Splunk logs",imgUrl:"public/img/plugins/splunk_logo_128.png"}),g({id:"grafana-oracle-datasource",name:"Oracle",description:"Visualize & explore Oracle SQL",imgUrl:"public/img/plugins/oracle.png"}),g({id:"grafana-dynatrace-datasource",name:"Dynatrace",description:"Visualize & explore Dynatrace data",imgUrl:"public/img/plugins/dynatrace.png"}),g({id:"grafana-servicenow-datasource",description:"ServiceNow integration & data source",name:"ServiceNow",imgUrl:"public/img/plugins/servicenow.svg"}),g({id:"grafana-datadog-datasource",description:"DataDog integration & data source",name:"DataDog",imgUrl:"public/img/plugins/datadog.png"}),g({id:"grafana-newrelic-datasource",description:"New Relic integration & data source",name:"New Relic",imgUrl:"public/img/plugins/newrelic.svg"}),g({id:"dlopes7-appdynamics-datasource",description:"AppDynamics integration & data source",name:"AppDynamics",imgUrl:"public/img/plugins/appdynamics.svg"})],o=!0,i=!1,c=void 0;try{for(var u,s=t[Symbol.iterator]();!(o=(u=s.next()).done);o=!0){var l=u.value;n[l.id]=l}}catch(e){i=!0,c=e}finally{try{o||null==s.return||s.return()}finally{if(i)throw c}}var f=!0,m=!1,b=void 0;try{for(var h,v=function(){var e=h.value;if(a.find((function(t){return t.id===e.id}))&&(e.category="enterprise"),e.info.links){var o=!0,i=!1,c=void 0;try{for(var u,s=e.info.links[Symbol.iterator]();!(o=(u=s.next()).done);o=!0){u.value.name="Learn more"}}catch(e){i=!0,c=e}finally{try{o||null==s.return||s.return()}finally{if(i)throw c}}}(t.find((function(t){return t.id===e.category}))||n.other).plugins.push(e),r[e.id]=e},y=e[Symbol.iterator]();!(f=(h=y.next()).done);f=!0)v()}catch(e){m=!0,b=e}finally{try{f||null==y.return||y.return()}finally{if(m)throw b}}var S=!0,w=!1,O=void 0;try{for(var D,j=t[Symbol.iterator]();!(S=(D=j.next()).done);S=!0){var x=D.value;if("cloud"===x.id&&x.plugins.push({id:"gcloud",name:"Grafana Cloud",type:d.PluginType.datasource,module:"phantom",baseUrl:"",info:{description:"Hosted Graphite, Prometheus and Loki",logos:{small:"public/img/grafana_icon.svg",large:"asd"},author:{name:"Grafana Labs"},links:[{url:"https://grafana.com/products/cloud/",name:"Learn more"}],screenshots:[],updated:"2019-05-10",version:"1.0.0"}}),"enterprise"===x.id){var E=!0,k=!1,P=void 0;try{for(var C,N=a[Symbol.iterator]();!(E=(C=N.next()).done);E=!0){var R=C.value;r[R.id]||x.plugins.push(R)}}catch(e){k=!0,P=e}finally{try{E||null==N.return||N.return()}finally{if(k)throw P}}}p(x.plugins)}}catch(e){w=!0,O=e}finally{try{S||null==j.return||j.return()}finally{if(w)throw O}}return t}function p(e){var t={prometheus:100,graphite:95,loki:90,mysql:80,jaeger:100,postgres:79,gcloud:-1};e.sort((function(e,n){var r=t[e.id]||0,a=t[n.id]||0;return r>a?-1:r<a?1:e.name>n.name?-1:1}))}function g(e){return{id:e.id,name:e.name,type:d.PluginType.datasource,module:"phantom",baseUrl:"",info:{description:e.description,logos:{small:e.imgUrl,large:e.imgUrl},author:{name:"Grafana Labs"},links:[{url:"https://grafana.com/grafana/plugins/"+e.id,name:"Install now"}],screenshots:[],updated:"2019-05-10",version:"1.0.0"}}}var m=n("frIo");function b(e,t,n,r,a,o,i){try{var c=e[o](i),u=c.value}catch(e){return void n(e)}c.done?t(u):Promise.resolve(u).then(r,a)}function h(e){return function(){var t=this,n=arguments;return new Promise((function(r,a){var o=e.apply(t,n);function i(e){b(o,r,a,i,c,"next",e)}function c(e){b(o,r,a,i,c,"throw",e)}i(void 0)}))}}n.d(t,"c",(function(){return v})),n.d(t,"g",(function(){return y})),n.d(t,"f",(function(){return S})),n.d(t,"d",(function(){return w})),n.d(t,"a",(function(){return O})),n.d(t,"e",(function(){return D})),n.d(t,"h",(function(){return j})),n.d(t,"b",(function(){return x}));var v=function(e){var t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:{loadDataSource:w,getDataSource:m.a,getDataSourceMeta:m.b,importDataSourcePlugin:s.b};return(function(){var n=h(regeneratorRuntime.mark((function n(r,a){var o,i,c;return regeneratorRuntime.wrap((function(n){for(;;)switch(n.prev=n.next){case 0:if(!isNaN(e)){n.next=3;break}return r(Object(l.g)(new Error("Invalid ID"))),n.abrupt("return");case 3:return n.prev=3,n.next=6,r(t.loadDataSource(e));case 6:if(!a().dataSourceSettings.plugin){n.next=8;break}return n.abrupt("return");case 8:return o=t.getDataSource(a().dataSources,e),i=t.getDataSourceMeta(a().dataSources,o.type),n.next=12,t.importDataSourcePlugin(i);case 12:c=n.sent,r(Object(l.h)(c)),n.next=20;break;case 16:n.prev=16,n.t0=n.catch(3),console.log("Failed to import plugin module",n.t0),r(Object(l.g)(n.t0));case 20:case"end":return n.stop()}}),n,null,[[3,16]])})));return function(e,t){return n.apply(this,arguments)}}())},y=function(e){var t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:{getDatasourceSrv:o.a,getBackendSrv:a.c};return(function(){var n=h(regeneratorRuntime.mark((function n(r,a){var o;return regeneratorRuntime.wrap((function(n){for(;;)switch(n.prev=n.next){case 0:return n.next=2,t.getDatasourceSrv().get(e);case 2:if((o=n.sent).testDatasource){n.next=5;break}return n.abrupt("return");case 5:r(Object(l.o)()),t.getBackendSrv().withNoBackendCache(h(regeneratorRuntime.mark((function e(){var t,n;return regeneratorRuntime.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.prev=0,e.next=3,o.testDatasource();case 3:t=e.sent,r(Object(l.p)(t)),e.next=12;break;case 7:e.prev=7,e.t0=e.catch(0),n="",n=e.t0.statusText?"HTTP Error "+e.t0.statusText:e.t0.message,r(Object(l.n)({message:n}));case 12:case"end":return e.stop()}}),e,null,[[0,7]])}))));case 7:case"end":return n.stop()}}),n)})));return function(e,t){return n.apply(this,arguments)}}())};function S(){return function(){var e=h(regeneratorRuntime.mark((function e(t){var n;return regeneratorRuntime.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.next=2,Object(a.c)().get("/api/datasources");case 2:n=e.sent,t(Object(l.e)(n));case 4:case"end":return e.stop()}}),e)})));return function(t){return e.apply(this,arguments)}}()}function w(e){return function(){var t=h(regeneratorRuntime.mark((function t(n){var r,o,d;return regeneratorRuntime.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:return t.next=2,Object(a.c)().get("/api/datasources/".concat(e));case 2:return r=t.sent,t.next=5,Object(u.a)(r.type);case 5:return o=t.sent,t.next=8,Object(s.b)(o);case 8:d=t.sent,n(Object(l.a)(r)),n(Object(l.b)(o)),n(Object(i.d)(Object(c.a)(r,d)));case 12:case"end":return t.stop()}}),t)})));return function(e){return t.apply(this,arguments)}}()}function O(e){return function(){var t=h(regeneratorRuntime.mark((function t(n,r){var o,c,u;return regeneratorRuntime.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:return t.next=2,n(S());case 2:return o=r().dataSources.dataSources,c={name:e.name,type:e.id,access:"proxy",isDefault:0===o.length},E(o,c.name)&&(c.name=k(o,c.name)),t.next=7,Object(a.c)().post("/api/datasources",c);case 7:u=t.sent,n(Object(i.c)({path:"/datasources/edit/".concat(u.id)}));case 9:case"end":return t.stop()}}),t)})));return function(e,n){return t.apply(this,arguments)}}()}function D(){return function(){var e=h(regeneratorRuntime.mark((function e(t){var n,r;return regeneratorRuntime.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return t(Object(l.c)()),e.next=3,Object(a.c)().get("/api/plugins",{enabled:1,type:"datasource"});case 3:n=e.sent,r=f(n),t(Object(l.d)({plugins:n,categories:r}));case 6:case"end":return e.stop()}}),e)})));return function(t){return e.apply(this,arguments)}}()}function j(e){return function(){var t=h(regeneratorRuntime.mark((function t(n){return regeneratorRuntime.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:return t.next=2,Object(a.c)().put("/api/datasources/".concat(e.id),e);case 2:return t.next=4,P();case 4:return t.abrupt("return",n(w(e.id)));case 5:case"end":return t.stop()}}),t)})));return function(e){return t.apply(this,arguments)}}()}function x(){return function(){var e=h(regeneratorRuntime.mark((function e(t,n){var r;return regeneratorRuntime.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return r=n().dataSources.dataSource,e.next=3,Object(a.c)().delete("/api/datasources/".concat(r.id));case 3:return e.next=5,P();case 5:t(Object(i.c)({path:"/datasources"}));case 6:case"end":return e.stop()}}),e)})));return function(t,n){return e.apply(this,arguments)}}()}function E(e,t){return e.filter((function(e){return e.name.toLowerCase()===t.toLowerCase()})).length>0}function k(e,t){for(;E(e,t);)t=C(t)?"".concat(R(t)).concat((n=N(t),isNaN(n)?1:n+1)):"".concat(t,"-1");var n;return t}function P(){return Object(a.c)().get("/api/frontend/settings").then((function(e){r.b.datasources=e.datasources,r.b.defaultDatasource=e.defaultDatasource,Object(o.a)().init()}))}function C(e){return e.endsWith("-",e.length-1)}function N(e){return parseInt(e.slice(-1),10)}function R(e){return e.slice(0,e.length-1)}},"7iUX":function(e,t,n){"use strict";var r=n("q1tI"),a=n.n(r),o=n("kDLi"),i=n("Obii"),c=n("kDDq");function u(){var e=function(e,t){t||(t=e.slice(0));return Object.freeze(Object.defineProperties(e,{raw:{value:Object.freeze(t)}}))}(["\n        margin-left: 16px;\n      "]);return u=function(){return e},e}t.a=function(e){var t=function(e){switch(e){case i.PluginState.alpha:return"Alpha Plugin: This plugin is a work in progress and updates may include breaking changes";case i.PluginState.beta:return"Beta Plugin: There could be bugs and minor breaking changes to this plugin"}return null}(e.state);return t?a.a.createElement(o.AlphaNotice,{state:e.state,text:t,className:Object(c.css)(u())}):null}},EREr:function(e,t,n){"use strict";n.d(t,"a",(function(){return p}));var r=n("q1tI"),a=n.n(r),o=n("LvDl"),i=n.n(o),c=n("t8hP");function u(e){return(u="function"==typeof Symbol&&"symbol"==typeof Symbol.iterator?function(e){return typeof e}:function(e){return e&&"function"==typeof Symbol&&e.constructor===Symbol&&e!==Symbol.prototype?"symbol":typeof e})(e)}function s(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}function l(e){return(l=Object.setPrototypeOf?Object.getPrototypeOf:function(e){return e.__proto__||Object.getPrototypeOf(e)})(e)}function d(e){if(void 0===e)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return e}function f(e,t){return(f=Object.setPrototypeOf||function(e,t){return e.__proto__=t,e})(e,t)}var p=function(e){function t(e){var n,r,a;return function(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}(this,t),r=this,a=l(t).call(this,e),(n=!a||"object"!==u(a)&&"function"!=typeof a?d(r):a).onModelChanged=function(e){n.props.onModelChange(e)},n.scopeProps={ctrl:{datasourceMeta:e.dataSourceMeta,current:i.a.cloneDeep(e.dataSource)},onModelChanged:n.onModelChanged},n.onModelChanged=n.onModelChanged.bind(d(n)),n}var n,r,o;return function(e,t){if("function"!=typeof t&&null!==t)throw new TypeError("Super expression must either be null or a function");e.prototype=Object.create(t&&t.prototype,{constructor:{value:e,writable:!0,configurable:!0}}),t&&f(e,t)}(t,e),n=t,(r=[{key:"componentDidMount",value:function(){var e=this.props.plugin;if(this.element&&!e.components.ConfigEditor){var t=Object(c.getAngularLoader)();this.component=t.load(this.element,this.scopeProps,'<plugin-component type="datasource-config-ctrl" />')}}},{key:"componentDidUpdate",value:function(e){this.props.plugin.components.ConfigEditor||this.props.dataSource===e.dataSource||(this.scopeProps.ctrl.current=i.a.cloneDeep(this.props.dataSource),this.component.digest())}},{key:"componentWillUnmount",value:function(){this.component&&this.component.destroy()}},{key:"render",value:function(){var e=this,t=this.props,n=t.plugin,r=t.dataSource;return n?a.a.createElement("div",{ref:function(t){return e.element=t}},n.components.ConfigEditor&&a.a.createElement(n.components.ConfigEditor,{options:r,onOptionsChange:this.onModelChanged})):null}}])&&s(n.prototype,r),o&&s(n,o),t}(r.PureComponent)},HUMP:function(e,t,n){"use strict";n.d(t,"a",(function(){return o})),n.d(t,"b",(function(){return i}));var r=n("Obii"),a=n("ZFWI");function o(e,t){var n=t.meta,r={img:n.info.logos.large,id:"datasource-"+e.id,subTitle:"Type: ".concat(n.name),url:"",text:e.name,breadcrumbs:[{title:"Data Sources",url:"datasources"}],children:[{active:!1,icon:"sliders-v-alt",id:"datasource-settings-".concat(e.id),text:"Settings",url:"datasources/edit/".concat(e.id,"/")}]};if(t.configPages){var o=!0,i=!1,c=void 0;try{for(var u,s=t.configPages[Symbol.iterator]();!(o=(u=s.next()).done);o=!0){var l=u.value;r.children.push({active:!1,text:l.title,icon:l.icon,url:"datasources/edit/".concat(e.id,"/?page=").concat(l.id),id:"datasource-page-".concat(l.id)})}}catch(e){i=!0,c=e}finally{try{o||null==s.return||s.return()}finally{if(i)throw c}}}return n.includes&&void 0!==n.includes.find((function(e){return"dashboard"===e.type}))&&r.children.push({active:!1,icon:"apps",id:"datasource-dashboards-".concat(e.id),text:"Dashboards",url:"datasources/edit/".concat(e.id,"/dashboards")}),a.b.licenseInfo.hasLicense&&r.children.push({active:!1,icon:"lock",id:"datasource-permissions-".concat(e.id),text:"Permissions",url:"datasources/edit/".concat(e.id,"/permissions")}),r}function i(e){var t,n=o({access:"",basicAuth:!1,basicAuthUser:"",basicAuthPassword:"",withCredentials:!1,database:"",id:1,isDefault:!1,jsonData:{authType:"credentials",defaultRegion:"eu-west-2"},name:"Loading",orgId:1,password:"",readOnly:!1,type:"Loading",typeLogoUrl:"public/img/icn-datasource.svg",url:"",user:""},{meta:{id:"1",type:r.PluginType.datasource,name:"",info:{author:{name:"",url:""},description:"",links:[{name:"",url:""}],logos:{large:"",small:""},screenshots:[],updated:"",version:""},includes:[],module:"",baseUrl:""}}),a=!0,i=!1,c=void 0;try{for(var u,s=n.children[Symbol.iterator]();!(a=(u=s.next()).done);a=!0){var l=u.value;if(l.id.indexOf(e)>0){l.active=!0,t=l;break}}}catch(e){i=!0,c=e}finally{try{a||null==s.return||s.return()}finally{if(i)throw c}}return{main:n,node:t}}},Jjpq:function(e,t,n){"use strict";var r=n("q1tI"),a=n.n(r),o=n("kDLi"),i=n("Csm0"),c=o.LegacyForms.Input,u=o.LegacyForms.Switch;t.a=function(e){var t=e.dataSourceName,n=e.isDefault,r=e.onDefaultChange,s=e.onNameChange;return a.a.createElement("div",{className:"gf-form-group","aria-label":"Datasource settings page basic settings"},a.a.createElement("div",{className:"gf-form-inline"},a.a.createElement("div",{className:"gf-form max-width-30",style:{marginRight:"3px"}},a.a.createElement(o.InlineFormLabel,{tooltip:"The name is used when you select the data source in panels. The Default data source is preselected in new panels."},"Name"),a.a.createElement(c,{className:"gf-form-input max-width-23",type:"text",value:t,placeholder:"Name",onChange:function(e){return s(e.target.value)},required:!0,"aria-label":i.selectors.pages.DataSource.name})),a.a.createElement(u,{label:"Default",checked:n,onChange:function(e){r(e.target.checked)}})))}},Klwq:function(e,t,n){"use strict";n.r(t),function(e){n.d(t,"DataSourceSettingsPage",(function(){return L}));var r=n("q1tI"),a=n.n(r),o=n("0cfB"),i=n("4qC0"),c=n.n(i),u=n("kDLi"),s=n("Csm0"),l=n("ZGyg"),d=n("EREr"),f=n("Jjpq"),p=n("/2Mn"),g=n("Xmxp"),m=n("frIo"),b=n("5BCB"),h=n("lzJ5"),v=n("X+V3"),y=n("GQ3c"),S=n("HUMP"),w=n("7iUX"),O=n("FFN/"),D=n("hBny");function j(e){return(j="function"==typeof Symbol&&"symbol"==typeof Symbol.iterator?function(e){return typeof e}:function(e){return e&&"function"==typeof Symbol&&e.constructor===Symbol&&e!==Symbol.prototype?"symbol":typeof e})(e)}function x(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function E(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?x(Object(n),!0).forEach((function(t){k(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):x(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function k(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function P(e,t,n,r,a,o,i){try{var c=e[o](i),u=c.value}catch(e){return void n(e)}c.done?t(u):Promise.resolve(u).then(r,a)}function C(e){return function(){var t=this,n=arguments;return new Promise((function(r,a){var o=e.apply(t,n);function i(e){P(o,r,a,i,c,"next",e)}function c(e){P(o,r,a,i,c,"throw",e)}i(void 0)}))}}function N(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}function R(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}function M(e,t){return!t||"object"!==j(t)&&"function"!=typeof t?function(e){if(void 0===e)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return e}(e):t}function I(e){return(I=Object.setPrototypeOf?Object.getPrototypeOf:function(e){return e.__proto__||Object.getPrototypeOf(e)})(e)}function T(e,t){return(T=Object.setPrototypeOf||function(e,t){return e.__proto__=t,e})(e,t)}var L=function(e){function t(){var e,n;N(this,t);for(var r=arguments.length,a=new Array(r),o=0;o<r;o++)a[o]=arguments[o];return(n=M(this,(e=I(t)).call.apply(e,[this].concat(a)))).onSubmit=function(){var e=C(regeneratorRuntime.mark((function e(t){return regeneratorRuntime.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return t.preventDefault(),e.next=3,n.props.updateDataSource(E({},n.props.dataSource));case 3:n.testDataSource();case 4:case"end":return e.stop()}}),e)})));return function(t){return e.apply(this,arguments)}}(),n.onTest=function(){var e=C(regeneratorRuntime.mark((function e(t){return regeneratorRuntime.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:t.preventDefault(),n.testDataSource();case 2:case"end":return e.stop()}}),e)})));return function(t){return e.apply(this,arguments)}}(),n.onDelete=function(){g.b.emit(y.CoreEvents.showConfirmModal,{title:"Delete",text:"Are you sure you want to delete this data source?",yesText:"Delete",icon:"trash-alt",onConfirm:function(){n.confirmDelete()}})},n.confirmDelete=function(){n.props.deleteDataSource()},n.onModelChange=function(e){n.props.dataSourceLoaded(e)},n}var n,r,o;return function(e,t){if("function"!=typeof t&&null!==t)throw new TypeError("Super expression must either be null or a function");e.prototype=Object.create(t&&t.prototype,{constructor:{value:e,writable:!0,configurable:!0}}),t&&T(e,t)}(t,e),n=t,(r=[{key:"componentDidMount",value:function(){var e=this.props;(0,e.initDataSourceSettings)(e.pageId)}},{key:"isReadOnly",value:function(){return!0===this.props.dataSource.readOnly}},{key:"renderIsReadOnlyMessage",value:function(){return a.a.createElement("div",{className:"grafana-info-box span8"},"This datasource was added by config and cannot be modified using the UI. Please contact your server admin to update this datasource.")}},{key:"testDataSource",value:function(){var e=this.props,t=e.dataSource,n=e.testDataSource;n(t.name)}},{key:"renderLoadError",value:function(e){var t=!1,n=e.toString();e.data?e.data.message&&(n=e.data.message):c()(e)&&(t=!0);var r={text:n,subTitle:"Data Source Error",icon:"exclamation-triangle"},o={node:r,main:r};return a.a.createElement(l.a,{navModel:o},a.a.createElement(l.a.Contents,null,a.a.createElement("div",null,a.a.createElement("div",{className:"gf-form-button-row"},t&&a.a.createElement("button",{type:"submit",className:"btn btn-danger",onClick:this.onDelete},"Delete"),a.a.createElement("a",{className:"btn btn-inverse",href:"datasources"},"Back")))))}},{key:"renderConfigPageBody",value:function(e){var t=this.props.plugin;if(!t||!t.configPages)return null;var n=!0,r=!1,o=void 0;try{for(var i,c=t.configPages[Symbol.iterator]();!(n=(i=c.next()).done);n=!0){var u=i.value;if(u.id===e)return a.a.createElement(u.body,{plugin:t,query:this.props.query})}}catch(e){r=!0,o=e}finally{try{n||null==c.return||c.return()}finally{if(r)throw o}}return a.a.createElement("div",null,"Page Not Found: ",e)}},{key:"renderSettings",value:function(){var e=this,t=this.props,n=t.dataSourceMeta,r=t.setDataSourceName,o=t.setIsDefault,i=t.dataSource,c=t.testingStatus,l=t.plugin;return a.a.createElement("form",{onSubmit:this.onSubmit},this.isReadOnly()&&this.renderIsReadOnlyMessage(),n.state&&a.a.createElement("div",{className:"gf-form"},a.a.createElement("label",{className:"gf-form-label width-10"},"Plugin state"),a.a.createElement("label",{className:"gf-form-label gf-form-label--transparent"},a.a.createElement(w.a,{state:n.state}))),a.a.createElement(f.a,{dataSourceName:i.name,isDefault:i.isDefault,onDefaultChange:function(e){return o(e)},onNameChange:function(e){return r(e)}}),l&&a.a.createElement(d.a,{plugin:l,dataSource:i,dataSourceMeta:n,onModelChange:this.onModelChange}),a.a.createElement("div",{className:"gf-form-group"},c&&c.message&&a.a.createElement("div",{className:"alert-".concat(c.status," alert"),"aria-label":s.selectors.pages.DataSource.alert},a.a.createElement("div",{className:"alert-icon"},"error"===c.status?a.a.createElement(u.Icon,{name:"exclamation-triangle"}):a.a.createElement(u.Icon,{name:"check"})),a.a.createElement("div",{className:"alert-body"},a.a.createElement("div",{className:"alert-title","aria-label":s.selectors.pages.DataSource.alertMessage},c.message)))),a.a.createElement(p.a,{onSubmit:function(t){return e.onSubmit(t)},isReadOnly:this.isReadOnly(),onDelete:this.onDelete,onTest:function(t){return e.onTest(t)}}))}},{key:"render",value:function(){var e=this.props,t=e.navModel,n=e.page,r=e.loadError;return r?this.renderLoadError(r):a.a.createElement(l.a,{navModel:t},a.a.createElement(l.a.Contents,{isLoading:!this.hasDataSource},this.hasDataSource&&a.a.createElement("div",null,n?this.renderConfigPageBody(n):this.renderSettings())))}},{key:"hasDataSource",get:function(){return this.props.dataSource.id>0}}])&&R(n.prototype,r),o&&R(n,o),t}(r.PureComponent);var U={deleteDataSource:b.b,loadDataSource:b.d,setDataSourceName:O.i,updateDataSource:b.h,setIsDefault:O.m,dataSourceLoaded:O.a,initDataSourceSettings:b.c,testDataSource:b.g};t.default=Object(o.hot)(e)(Object(D.a)((function(e){var t=Object(v.c)(e.location),n=Object(m.a)(e.dataSources,t),r=e.location.query.page,a=e.dataSourceSettings,o=a.plugin,i=a.loadError,c=a.testingStatus;return{navModel:Object(h.a)(e.navIndex,r?"datasource-page-".concat(r):"datasource-settings-".concat(t),Object(S.b)("settings")),dataSource:Object(m.a)(e.dataSources,t),dataSourceMeta:Object(m.b)(e.dataSources,n.type),pageId:t,query:e.location.query,page:r,plugin:o,loadError:i,testingStatus:c}}),U,(function(e){return e.dataSourceSettings}))(L))}.call(this,n("3UD+")(e))},frIo:function(e,t,n){"use strict";n.d(t,"d",(function(){return r})),n.d(t,"c",(function(){return a})),n.d(t,"a",(function(){return o})),n.d(t,"b",(function(){return i})),n.d(t,"g",(function(){return c})),n.d(t,"f",(function(){return u})),n.d(t,"e",(function(){return s}));var r=function(e){var t=new RegExp(e.searchQuery,"i");return e.dataSources.filter((function(e){return t.test(e.name)||t.test(e.database)}))},a=function(e){var t=new RegExp(e.dataSourceTypeSearchQuery,"i");return e.plugins.filter((function(e){return t.test(e.name)}))},o=function(e,t){return e.dataSource.id===parseInt(t,10)?e.dataSource:{}},i=function(e,t){return e.dataSourceMeta.id===t?e.dataSourceMeta:{}},c=function(e){return e.searchQuery},u=function(e){return e.layoutMode},s=function(e){return e.dataSourcesCount}},hBny:function(e,t,n){"use strict";n.d(t,"a",(function(){return s}));var r=n("/MKj"),a=n("zVNn"),o=n("q1tI"),i=n.n(o),c=n("2mql"),u=n.n(c),s=function(e,t,n){return function(c){var s=Object(r.connect)(e,t)(c),l=function(e){var t=Object(r.useDispatch)();return Object(o.useEffect)((function(){return function(){t(Object(a.a)({stateSelector:n}))}}),[]),i.a.createElement(s,e)};return l.displayName="ConnectWithCleanUp(".concat(s.displayName,")"),u()(l,c),l}}}}]);
//# sourceMappingURL=DataSourceSettingsPage.c8856b8b39626543db12.js.map