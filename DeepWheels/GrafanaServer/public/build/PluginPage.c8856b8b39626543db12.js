(window.webpackJsonp=window.webpackJsonp||[]).push([[2],{"3Q5D":function(e,t,n){"use strict";n.d(t,"a",(function(){return g}));var r=n("q1tI"),a=n.n(r),o=n("zdiy"),i=n.n(o),c=n("t8hP"),l=n("HJRA"),u=n("vHOe"),s=n("Obii");function p(e){return(p="function"==typeof Symbol&&"symbol"==typeof Symbol.iterator?function(e){return typeof e}:function(e){return e&&"function"==typeof Symbol&&e.constructor===Symbol&&e!==Symbol.prototype?"symbol":typeof e})(e)}function f(e,t,n,r,a,o,i){try{var c=e[o](i),l=c.value}catch(e){return void n(e)}c.done?t(l):Promise.resolve(l).then(r,a)}function d(e){return function(e){if(Array.isArray(e)){for(var t=0,n=new Array(e.length);t<e.length;t++)n[t]=e[t];return n}}(e)||function(e){if(Symbol.iterator in Object(e)||"[object Arguments]"===Object.prototype.toString.call(e))return Array.from(e)}(e)||function(){throw new TypeError("Invalid attempt to spread non-iterable instance")}()}function m(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}function b(e,t){return!t||"object"!==p(t)&&"function"!=typeof t?function(e){if(void 0===e)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return e}(e):t}function h(e){return(h=Object.setPrototypeOf?Object.getPrototypeOf:function(e){return e.__proto__||Object.getPrototypeOf(e)})(e)}function y(e,t){return(y=Object.setPrototypeOf||function(e,t){return e.__proto__=t,e})(e,t)}var g=function(e){function t(e){var n;return function(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}(this,t),(n=b(this,h(t).call(this,e))).importAll=function(){n.importNext(0)},n.importNext=function(e){var t=n.state.dashboards;return n.import(t[e],!0).then((function(){return e+1<t.length?new Promise((function(t){setTimeout((function(){n.importNext(e+1).then((function(){t()}))}),500)})):Promise.resolve()}))},n.import=function(e,t){var r=n.props,a=r.plugin,o=r.datasource,u={pluginId:a.id,path:e.path,overwrite:t,inputs:[]};return o&&u.inputs.push({name:"*",type:"datasource",pluginId:o.meta.id,value:o.name}),Object(c.getBackendSrv)().post("/api/dashboards/import",u).then((function(t){l.a.emit(s.AppEvents.alertSuccess,["Dashboard Imported",e.title]),i()(e,t),n.setState({dashboards:d(n.state.dashboards)})}))},n.remove=function(e){Object(c.getBackendSrv)().delete("/api/dashboards/"+e.importedUri).then((function(){e.imported=!1,n.setState({dashboards:d(n.state.dashboards)})}))},n.state={loading:!0,dashboards:[]},n}var n,r,o,p,g;return function(e,t){if("function"!=typeof t&&null!==t)throw new TypeError("Super expression must either be null or a function");e.prototype=Object.create(t&&t.prototype,{constructor:{value:e,writable:!0,configurable:!0}}),t&&y(e,t)}(t,e),n=t,(r=[{key:"componentDidMount",value:(p=regeneratorRuntime.mark((function e(){var t,n=this;return regeneratorRuntime.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:t=this.props.plugin.id,Object(c.getBackendSrv)().get("/api/plugins/".concat(t,"/dashboards")).then((function(e){n.setState({dashboards:e,loading:!1})}));case 2:case"end":return e.stop()}}),e,this)})),g=function(){var e=this,t=arguments;return new Promise((function(n,r){var a=p.apply(e,t);function o(e){f(a,n,r,o,i,"next",e)}function i(e){f(a,n,r,o,i,"throw",e)}o(void 0)}))},function(){return g.apply(this,arguments)})},{key:"render",value:function(){var e=this.state,t=e.loading,n=e.dashboards;return t?a.a.createElement("div",null,"loading..."):n&&n.length?a.a.createElement("div",{className:"gf-form-group"},a.a.createElement(u.a,{dashboards:n,onImport:this.import,onRemove:this.remove})):a.a.createElement("div",null,"No dashboards are included with this plugin")}}])&&m(n.prototype,r),o&&m(n,o),t}(r.PureComponent)},"4HI1":function(e,t,n){"use strict";n.d(t,"a",(function(){return g}));var r=n("q1tI"),a=n.n(r),o=n("BkRI"),i=n.n(o),c=n("zdiy"),l=n.n(c),u=n("kDLi"),s=n("Obii"),p=n("t8hP"),f=n("kDDq");function d(e){return(d="function"==typeof Symbol&&"symbol"==typeof Symbol.iterator?function(e){return typeof e}:function(e){return e&&"function"==typeof Symbol&&e.constructor===Symbol&&e!==Symbol.prototype?"symbol":typeof e})(e)}function m(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}function b(e,t){return!t||"object"!==d(t)&&"function"!=typeof t?function(e){if(void 0===e)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return e}(e):t}function h(e){return(h=Object.setPrototypeOf?Object.getPrototypeOf:function(e){return e.__proto__||Object.getPrototypeOf(e)})(e)}function y(e,t){return(y=Object.setPrototypeOf||function(e,t){return e.__proto__=t,e})(e,t)}var g=function(e){function t(e){var n;return function(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}(this,t),(n=b(this,h(t).call(this,e))).preUpdateHook=function(){return Promise.resolve()},n.postUpdateHook=function(){return Promise.resolve()},n.update=function(){var e=n.model.id;n.preUpdateHook().then((function(){var t=l()({enabled:n.model.enabled,pinned:n.model.pinned,jsonData:n.model.jsonData,secureJsonData:n.model.secureJsonData},{});return Object(p.getBackendSrv)().post("/api/plugins/".concat(e,"/settings"),t)})).then(n.postUpdateHook).then((function(e){window.location.href=window.location.href}))},n.setPreUpdateHook=function(e){n.preUpdateHook=e},n.setPostUpdateHook=function(e){n.postUpdateHook=e},n.importDashboards=function(){return Object(s.deprecationWarning)("AppConfig","importDashboards()"),Promise.resolve()},n.enable=function(){n.model.enabled=!0,n.model.pinned=!0,n.update()},n.disable=function(){n.model.enabled=!1,n.model.pinned=!1,n.update()},n.state={angularCtrl:null,refresh:0},n}var n,r,o;return function(e,t){if("function"!=typeof t&&null!==t)throw new TypeError("Super expression must either be null or a function");e.prototype=Object.create(t&&t.prototype,{constructor:{value:e,writable:!0,configurable:!0}}),t&&y(e,t)}(t,e),n=t,(r=[{key:"componentDidMount",value:function(){var e=this;setTimeout((function(){e.setState({refresh:e.state.refresh+1})}),5)}},{key:"componentDidUpdate",value:function(e){if(this.element&&!this.state.angularCtrl){this.model=i()(this.props.app.meta);var t={ctrl:this},n=Object(p.getAngularLoader)().load(this.element,t,'<plugin-component type="app-config-ctrl"></plugin-component>');this.setState({angularCtrl:n})}}},{key:"render",value:function(){var e=this,t=this.model,n=Object(f.css)({marginRight:"8px"});return a.a.createElement("div",null,a.a.createElement("div",{ref:function(t){return e.element=t}}),a.a.createElement("br",null),a.a.createElement("br",null),t&&a.a.createElement("div",{className:"gf-form"},!t.enabled&&a.a.createElement(u.Button,{variant:"primary",onClick:this.enable,className:n},"Enable"),t.enabled&&a.a.createElement(u.Button,{variant:"primary",onClick:this.update,className:n},"Update"),t.enabled&&a.a.createElement(u.Button,{variant:"destructive",onClick:this.disable,className:n},"Disable")))}}])&&m(n.prototype,r),o&&m(n,o),t}(r.PureComponent)},"OG+f":function(e,t,n){"use strict";n.d(t,"a",(function(){return d}));var r=n("q1tI"),a=n.n(r),o=n("Obii"),i=n("t8hP");function c(e){return(c="function"==typeof Symbol&&"symbol"==typeof Symbol.iterator?function(e){return typeof e}:function(e){return e&&"function"==typeof Symbol&&e.constructor===Symbol&&e!==Symbol.prototype?"symbol":typeof e})(e)}function l(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}function u(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}function s(e,t){return!t||"object"!==c(t)&&"function"!=typeof t?function(e){if(void 0===e)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return e}(e):t}function p(e){return(p=Object.setPrototypeOf?Object.getPrototypeOf:function(e){return e.__proto__||Object.getPrototypeOf(e)})(e)}function f(e,t){return(f=Object.setPrototypeOf||function(e,t){return e.__proto__=t,e})(e,t)}var d=function(e){function t(){var e,n;l(this,t);for(var r=arguments.length,a=new Array(r),c=0;c<r;c++)a[c]=arguments[c];return(n=s(this,(e=p(t)).call.apply(e,[this].concat(a)))).state={isError:!1,isLoading:!1,help:""},n.loadHelp=function(){var e=n.props,t=e.plugin,r=e.type;n.setState({isLoading:!0}),Object(i.getBackendSrv)().get("/api/plugins/".concat(t.id,"/markdown/").concat(r)).then((function(e){var t=Object(o.renderMarkdown)(e);""===e&&"help"===r?n.setState({isError:!1,isLoading:!1,help:n.constructPlaceholderInfo()}):n.setState({isError:!1,isLoading:!1,help:t})})).catch((function(){n.setState({isError:!0,isLoading:!1})}))},n}var n,r,c;return function(e,t){if("function"!=typeof t&&null!==t)throw new TypeError("Super expression must either be null or a function");e.prototype=Object.create(t&&t.prototype,{constructor:{value:e,writable:!0,configurable:!0}}),t&&f(e,t)}(t,e),n=t,(r=[{key:"componentDidMount",value:function(){this.loadHelp()}},{key:"constructPlaceholderInfo",value:function(){return"No plugin help or readme markdown file was found"}},{key:"render",value:function(){this.props.type;var e=this.state,t=e.isError,n=e.isLoading,r=e.help;return n?a.a.createElement("h2",null,"Loading help..."):t?a.a.createElement("h3",null,"'Error occurred when loading help'"):a.a.createElement("div",{className:"markdown-html",dangerouslySetInnerHTML:{__html:r}})}}])&&u(n.prototype,r),c&&u(n,c),t}(r.PureComponent)},kYMR:function(e,t,n){"use strict";n.r(t),function(e){n.d(t,"getLoadingNav",(function(){return D}));var r=n("q1tI"),a=n.n(r),o=n("0cfB"),i=n("/MKj"),c=n("J2m7"),l=n.n(c),u=n("Obii"),s=n("GQ3c"),p=n("kDLi"),f=n("ZGyg"),d=n("okuo"),m=n("Vw/f"),b=n("lJbD"),h=n("OG+f"),y=n("4HI1"),g=n("3Q5D"),v=n("HJRA"),E=n("ZFWI");function O(e){return(O="function"==typeof Symbol&&"symbol"==typeof Symbol.iterator?function(e){return typeof e}:function(e){return e&&"function"==typeof Symbol&&e.constructor===Symbol&&e!==Symbol.prototype?"symbol":typeof e})(e)}function w(e,t,n,r,a,o,i){try{var c=e[o](i),l=c.value}catch(e){return void n(e)}c.done?t(l):Promise.resolve(l).then(r,a)}function k(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}function P(e,t){return!t||"object"!==O(t)&&"function"!=typeof t?function(e){if(void 0===e)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return e}(e):t}function j(e){return(j=Object.setPrototypeOf?Object.getPrototypeOf:function(e){return e.__proto__||Object.getPrototypeOf(e)})(e)}function S(e,t){return(S=Object.setPrototypeOf||function(e,t){return e.__proto__=t,e})(e,t)}function N(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function I(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?N(Object(n),!0).forEach((function(t){_(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):N(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function _(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function D(){var e={text:"Loading...",icon:"icon-gf icon-gf-panel"};return{node:e,main:e}}function x(e){return Object(d.a)(e).then((function(t){return t.type===u.PluginType.app?Object(m.a)(t):t.type===u.PluginType.datasource?Object(m.b)(t):t.type===u.PluginType.panel?Object(m.c)(e).then((function(t){return Object(d.a)(e).then((function(e){return t.meta=I({},e,{},t.meta),t}))})):t.type===u.PluginType.renderer?Promise.resolve({meta:t}):Promise.reject("Unknown Plugin type: "+t.type)}))}var C=function(e){function t(e){var n;return function(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}(this,t),(n=P(this,j(t).call(this,e))).showUpdateInfo=function(){v.a.emit(s.CoreEvents.showModal,{src:"public/app/features/plugins/partials/update_instructions.html",model:n.state.plugin.meta})},n.state={loading:!0,nav:D(),defaultPage:"readme"},n}var n,r,o,i,c;return function(e,t){if("function"!=typeof t&&null!==t)throw new TypeError("Super expression must either be null or a function");e.prototype=Object.create(t&&t.prototype,{constructor:{value:e,writable:!0,configurable:!0}}),t&&S(e,t)}(t,e),n=t,(r=[{key:"componentDidMount",value:(i=regeneratorRuntime.mark((function e(){var t,n,r,a,o,i,c,l,u,s;return regeneratorRuntime.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return t=this.props,n=t.pluginId,r=t.path,a=t.query,o=t.$contextSrv,i=E.a.appSubUrl,e.next=4,x(n);case 4:if(c=e.sent){e.next=8;break}return this.setState({loading:!1,nav:Object(b.a)()}),e.abrupt("return");case 8:l=T(c,i,r,a,o.hasRole("Admin")),u=l.defaultPage,s=l.nav,this.setState({loading:!1,plugin:c,defaultPage:u,nav:s});case 10:case"end":return e.stop()}}),e,this)})),c=function(){var e=this,t=arguments;return new Promise((function(n,r){var a=i.apply(e,t);function o(e){w(a,n,r,o,c,"next",e)}function c(e){w(a,n,r,o,c,"throw",e)}o(void 0)}))},function(){return c.apply(this,arguments)})},{key:"componentDidUpdate",value:function(e){var t=e.query.page,n=this.props.query.page;if(t!==n){var r=this.state,a=r.nav,o=r.defaultPage,i=I({},a.node,{children:L(n,a.node.children,o)});this.setState({nav:{node:i,main:i}})}}},{key:"renderBody",value:function(){var e=this.props.query,t=this.state,n=t.plugin,r=t.nav;if(!n)return a.a.createElement(p.Alert,{severity:s.AppNotificationSeverity.Error,title:"Plugin Not Found"});var o=r.main.children.find((function(e){return e.active}));if(o){if(n.configPages){var i=!0,c=!1,l=void 0;try{for(var f,d=n.configPages[Symbol.iterator]();!(i=(f=d.next()).done);i=!0){var m=f.value;if(m.id===o.id)return a.a.createElement(m.body,{plugin:n,query:e})}}catch(e){c=!0,l=e}finally{try{i||null==d.return||d.return()}finally{if(c)throw l}}}if(n.meta.type===u.PluginType.app){if("dashboards"===o.id)return a.a.createElement(g.a,{plugin:n.meta});if("config"===o.id&&n.angularConfigCtrl)return a.a.createElement(y.a,{app:n})}}return a.a.createElement(h.a,{plugin:n.meta,type:"help"})}},{key:"renderVersionInfo",value:function(e){return e.info.version?a.a.createElement("section",{className:"page-sidebar-section"},a.a.createElement("h4",null,"Version"),a.a.createElement("span",null,e.info.version),e.hasUpdate&&a.a.createElement("div",null,a.a.createElement(p.Tooltip,{content:e.latestVersion,theme:"info",placement:"top"},a.a.createElement("a",{href:"#",onClick:this.showUpdateInfo},"Update Available!")))):null}},{key:"renderSidebarIncludeBody",value:function(e){if(e.type===u.PluginIncludeType.page){var t=this.state.plugin.meta.id,n=e.name.toLowerCase().replace(" ","-");return a.a.createElement("a",{href:"plugins/".concat(t,"/page/").concat(n)},a.a.createElement("i",{className:R(e.type)}),e.name)}return a.a.createElement(a.a.Fragment,null,a.a.createElement("i",{className:R(e.type)}),e.name)}},{key:"renderSidebarIncludes",value:function(e){var t=this;return e&&e.length?a.a.createElement("section",{className:"page-sidebar-section"},a.a.createElement("h4",null,"Includes"),a.a.createElement("ul",{className:"ui-list plugin-info-list"},e.map((function(e){return a.a.createElement("li",{className:"plugin-info-list-item",key:e.name},t.renderSidebarIncludeBody(e))})))):null}},{key:"renderSidebarDependencies",value:function(e){return e?a.a.createElement("section",{className:"page-sidebar-section"},a.a.createElement("h4",null,"Dependencies"),a.a.createElement("ul",{className:"ui-list plugin-info-list"},a.a.createElement("li",{className:"plugin-info-list-item"},a.a.createElement("img",{src:"public/img/grafana_icon.svg"}),"Grafana ",e.grafanaVersion),e.plugins&&e.plugins.map((function(e){return a.a.createElement("li",{className:"plugin-info-list-item",key:e.name},a.a.createElement("i",{className:R(e.type)}),e.name," ",e.version)})))):null}},{key:"renderSidebarLinks",value:function(e){return e.links&&e.links.length?a.a.createElement("section",{className:"page-sidebar-section"},a.a.createElement("h4",null,"Links"),a.a.createElement("ul",{className:"ui-list"},e.links.map((function(e){return a.a.createElement("li",{key:e.url},a.a.createElement("a",{href:e.url,className:"external-link",target:"_blank",rel:"noopener"},e.name))})))):null}},{key:"render",value:function(){var e=this.state,t=e.loading,n=e.nav,r=e.plugin,o=this.props.$contextSrv.hasRole("Admin");return a.a.createElement(f.a,{navModel:n},a.a.createElement(f.a.Contents,{isLoading:t},!t&&a.a.createElement("div",{className:"sidebar-container"},a.a.createElement("div",{className:"sidebar-content"},r.loadError&&a.a.createElement(p.Alert,{severity:s.AppNotificationSeverity.Error,title:"Error Loading Plugin",children:a.a.createElement(a.a.Fragment,null,"Check the server startup logs for more information. ",a.a.createElement("br",null),"If this plugin was loaded from git, make sure it was compiled.")}),this.renderBody()),a.a.createElement("aside",{className:"page-sidebar"},r&&a.a.createElement("section",{className:"page-sidebar-section"},this.renderVersionInfo(r.meta),o&&this.renderSidebarIncludes(r.meta.includes),this.renderSidebarDependencies(r.meta.dependencies),this.renderSidebarLinks(r.meta.info))))))}}])&&k(n.prototype,r),o&&k(n,o),t}(r.PureComponent);function T(e,t,n,r,a){var o,i=e.meta,c=[];if(c.push({text:"Readme",icon:"fa fa-fw fa-file-text-o",url:"".concat(t).concat(n,"?page=").concat("readme"),id:"readme"}),a&&i.type===u.PluginType.app){if(e.angularConfigCtrl&&(c.push({text:"Config",icon:"gicon gicon-cog",url:"".concat(t).concat(n,"?page=").concat("config"),id:"config"}),o="config"),e.configPages){var s=!0,p=!1,f=void 0;try{for(var d,m=e.configPages[Symbol.iterator]();!(s=(d=m.next()).done);s=!0){var b=d.value;c.push({text:b.title,icon:b.icon,url:"".concat(t).concat(n,"?page=").concat(b.id),id:b.id}),o||(o=b.id)}}catch(e){p=!0,f=e}finally{try{s||null==m.return||m.return()}finally{if(p)throw f}}}l()(i.includes,{type:u.PluginIncludeType.dashboard})&&c.push({text:"Dashboards",icon:"gicon gicon-dashboard",url:"".concat(t).concat(n,"?page=").concat("dashboards"),id:"dashboards"})}o||(o=c[0].id);var h={text:i.name,img:i.info.logos.large,subTitle:i.info.author.name,breadcrumbs:[{title:"Plugins",url:"plugins"}],url:"".concat(t).concat(n),children:L(r.page,c,o)};return{defaultPage:o,nav:{node:h,main:h}}}function L(e,t,n){var r=!1,a=e||n,o=t.map((function(e){var t=!r&&a===e.id;return t&&(r=!0),I({},e,{active:t})}));return r||(o[0].active=!0),o}function R(e){switch(e){case"datasource":return"gicon gicon-datasources";case"panel":return"icon-gf icon-gf-panel";case"app":return"icon-gf icon-gf-apps";case"page":return"icon-gf icon-gf-endpoint-tiny";case"dashboard":return"gicon gicon-dashboard";default:return"icon-gf icon-gf-apps"}}t.default=Object(o.hot)(e)(Object(i.connect)((function(e){return{pluginId:e.location.routeParams.pluginId,query:e.location.query,path:e.location.path}}))(C))}.call(this,n("3UD+")(e))},qbnB:function(e,t,n){var r=n("juv8"),a=n("LsHQ"),o=n("mTTR"),i=a((function(e,t){r(t,o(t),e)}));e.exports=i},vHOe:function(e,t,n){"use strict";var r=n("q1tI"),a=n.n(r),o=n("kDLi");t.a=function(e){var t=e.dashboards,n=e.onImport,r=e.onRemove;return a.a.createElement("table",{className:"filter-table"},a.a.createElement("tbody",null,t.map((function(e,t){return a.a.createElement("tr",{key:"".concat(e.dashboardId,"-").concat(t)},a.a.createElement("td",{className:"width-1"},a.a.createElement(o.Icon,{name:"apps"})),a.a.createElement("td",null,e.imported?a.a.createElement("a",{href:e.importedUrl},e.title):a.a.createElement("span",null,e.title)),a.a.createElement("td",{style:{textAlign:"right"}},e.imported?a.a.createElement("button",{className:"btn btn-secondary btn-small",onClick:function(){return n(e,!0)}},function(e){return e.revision!==e.importedRevision?"Update":"Re-import"}(e)):a.a.createElement("button",{className:"btn btn-secondary btn-small",onClick:function(){return n(e,!1)}},"Import"),e.imported&&a.a.createElement("button",{className:"btn btn-danger btn-small",onClick:function(){return r(e)}},a.a.createElement(o.Icon,{name:"trash-alt"}))))}))))}},zdiy:function(e,t,n){e.exports=n("qbnB")}}]);
//# sourceMappingURL=PluginPage.c8856b8b39626543db12.js.map