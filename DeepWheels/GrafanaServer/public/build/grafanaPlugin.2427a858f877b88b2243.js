(window.webpackJsonp=window.webpackJsonp||[]).push([[53],{gcd9:function(t,n,e){"use strict";e.r(n);var r=e("LvDl"),o=e.n(r),a=e("t8hP"),i=e("Obii"),u=e("5kRJ");function c(t){return(c="function"==typeof Symbol&&"symbol"==typeof Symbol.iterator?function(t){return typeof t}:function(t){return t&&"function"==typeof Symbol&&t.constructor===Symbol&&t!==Symbol.prototype?"symbol":typeof t})(t)}function f(t,n){for(var e=0;e<n.length;e++){var r=n[e];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(t,r.key,r)}}function l(t,n){return!n||"object"!==c(n)&&"function"!=typeof n?function(t){if(void 0===t)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return t}(t):n}function s(t){return(s=Object.setPrototypeOf?Object.getPrototypeOf:function(t){return t.__proto__||Object.getPrototypeOf(t)})(t)}function y(t,n){return(y=Object.setPrototypeOf||function(t,n){return t.__proto__=n,t})(t,n)}var p=function(t){function n(t){return function(t,n){if(!(t instanceof n))throw new TypeError("Cannot call a class as a function")}(this,n),l(this,s(n).call(this,t))}var e,r,i;return n.$inject=["instanceSettings"],function(t,n){if("function"!=typeof n&&null!==n)throw new TypeError("Super expression must either be null or a function");t.prototype=Object.create(n&&n.prototype,{constructor:{value:t,writable:!0,configurable:!0}}),n&&y(t,n)}(n,t),e=n,(r=[{key:"query",value:function(t){return Object(a.getBackendSrv)().get("/api/tsdb/testdata/random-walk",{from:t.range.from.valueOf(),to:t.range.to.valueOf(),intervalMs:t.intervalMs,maxDataPoints:t.maxDataPoints}).then((function(t){var n=[];return t.results&&o.a.forEach(t.results,(function(t){var e=!0,r=!1,o=void 0;try{for(var a,i=t.series[Symbol.iterator]();!(e=(a=i.next()).done);e=!0){var u=a.value;n.push({target:u.name,datapoints:u.points})}}catch(t){r=!0,o=t}finally{try{e||null==i.return||i.return()}finally{if(r)throw o}}})),{data:n}}))}},{key:"metricFindQuery",value:function(t){return Promise.resolve([])}},{key:"annotationQuery",value:function(t){var n,e={from:t.range.from.valueOf(),to:t.range.to.valueOf(),limit:t.annotation.limit,tags:t.annotation.tags,matchAny:t.annotation.matchAny};if("dashboard"===t.annotation.type){if(!t.dashboard.id)return Promise.resolve([]);e.dashboardId=t.dashboard.id,delete e.tags}else{var r=function(){if(!o.a.isArray(t.annotation.tags)||0===t.annotation.tags.length)return{v:Promise.resolve([])};var n=[],r=!0,a=!1,i=void 0;try{for(var c,f=e.tags[Symbol.iterator]();!(r=(c=f.next()).done);r=!0){var l=c.value,s=u.b.replace(l,{},(function(t){return"string"==typeof t?t:t.join("__delimiter__")})),y=!0,p=!1,b=void 0;try{for(var d,v=s.split("__delimiter__")[Symbol.iterator]();!(y=(d=v.next()).done);y=!0){var h=d.value;n.push(h)}}catch(t){p=!0,b=t}finally{try{y||null==v.return||v.return()}finally{if(p)throw b}}}}catch(t){a=!0,i=t}finally{try{r||null==f.return||f.return()}finally{if(a)throw i}}e.tags=n}();if("object"===c(r))return r.v}return Object(a.getBackendSrv)().get("/api/annotations",e,"grafana-data-source-annotations-".concat(t.annotation.name,"-").concat(null===(n=t.dashboard)||void 0===n?void 0:n.id))}},{key:"testDatasource",value:function(){return Promise.resolve()}}])&&f(e.prototype,r),i&&f(e,i),n}(i.DataSourceApi),b=e("LzXI");function d(t){return(d="function"==typeof Symbol&&"symbol"==typeof Symbol.iterator?function(t){return typeof t}:function(t){return t&&"function"==typeof Symbol&&t.constructor===Symbol&&t!==Symbol.prototype?"symbol":typeof t})(t)}function v(t,n){if(!(t instanceof n))throw new TypeError("Cannot call a class as a function")}function h(t,n){return!n||"object"!==d(n)&&"function"!=typeof n?function(t){if(void 0===t)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return t}(t):n}function m(t){return(m=Object.setPrototypeOf?Object.getPrototypeOf:function(t){return t.__proto__||Object.getPrototypeOf(t)})(t)}function g(t,n){return(g=Object.setPrototypeOf||function(t,n){return t.__proto__=n,t})(t,n)}e.d(n,"QueryCtrl",(function(){return O})),e.d(n,"AnnotationsQueryCtrl",(function(){return w})),e.d(n,"GrafanaDatasource",(function(){return p})),e.d(n,"Datasource",(function(){return p}));var O=function(t){function n(){return v(this,n),h(this,m(n).apply(this,arguments))}return function(t,n){if("function"!=typeof n&&null!==n)throw new TypeError("Super expression must either be null or a function");t.prototype=Object.create(n&&n.prototype,{constructor:{value:t,writable:!0,configurable:!0}}),n&&g(t,n)}(n,t),n}(b.QueryCtrl);O.templateUrl="partials/query.editor.html";var w=function t(){v(this,t),this.types=[{text:"Dashboard",value:"dashboard"},{text:"Tags",value:"tags"}],this.annotation.type=this.annotation.type||"tags",this.annotation.limit=this.annotation.limit||100};w.templateUrl="partials/annotations.editor.html"}}]);
//# sourceMappingURL=grafanaPlugin.2427a858f877b88b2243.js.map