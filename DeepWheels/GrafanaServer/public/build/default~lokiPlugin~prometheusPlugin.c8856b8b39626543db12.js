(window.webpackJsonp=window.webpackJsonp||[]).push([[4],{HwNo:function(e,t,n){"use strict";e.exports=function(e){e.prototype[Symbol.iterator]=regeneratorRuntime.mark((function e(){var t;return regeneratorRuntime.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:t=this.head;case 1:if(!t){e.next=7;break}return e.next=4,t.value;case 4:t=t.next,e.next=1;break;case 7:case"end":return e.stop()}}),e,this)}))}},HyWp:function(e,t,n){"use strict";function a(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}function r(e,t){for(var n=0;n<t.length;n++){var a=t[n];a.enumerable=a.enumerable||!1,a.configurable=!0,"value"in a&&(a.writable=!0),Object.defineProperty(e,a.key,a)}}var i=n("XPeR"),o=Symbol("max"),s=Symbol("length"),l=Symbol("lengthCalculator"),u=Symbol("allowStale"),c=Symbol("maxAge"),h=Symbol("dispose"),v=Symbol("noDisposeOnSet"),m=Symbol("lruList"),d=Symbol("cache"),f=Symbol("updateAgeOnGet"),p=function(){return 1},g=function(){function e(t){if(a(this,e),"number"==typeof t&&(t={max:t}),t||(t={}),t.max&&("number"!=typeof t.max||t.max<0))throw new TypeError("max must be a non-negative number");this[o]=t.max||1/0;var n=t.length||p;if(this[l]="function"!=typeof n?p:n,this[u]=t.stale||!1,t.maxAge&&"number"!=typeof t.maxAge)throw new TypeError("maxAge must be a number");this[c]=t.maxAge||0,this[h]=t.dispose,this[v]=t.noDisposeOnSet||!1,this[f]=t.updateAgeOnGet||!1,this.reset()}var t,n,g;return t=e,(n=[{key:"rforEach",value:function(e,t){t=t||this;for(var n=this[m].tail;null!==n;){var a=n.prev;T(this,e,n,t),n=a}}},{key:"forEach",value:function(e,t){t=t||this;for(var n=this[m].head;null!==n;){var a=n.next;T(this,e,n,t),n=a}}},{key:"keys",value:function(){return this[m].toArray().map((function(e){return e.key}))}},{key:"values",value:function(){return this[m].toArray().map((function(e){return e.value}))}},{key:"reset",value:function(){var e=this;this[h]&&this[m]&&this[m].length&&this[m].forEach((function(t){return e[h](t.key,t.value)})),this[d]=new Map,this[m]=new i,this[s]=0}},{key:"dump",value:function(){var e=this;return this[m].map((function(t){return!y(e,t)&&{k:t.key,v:t.value,e:t.now+(t.maxAge||0)}})).toArray().filter((function(e){return e}))}},{key:"dumpLru",value:function(){return this[m]}},{key:"set",value:function(e,t,n){if((n=n||this[c])&&"number"!=typeof n)throw new TypeError("maxAge must be a number");var a=n?Date.now():0,r=this[l](t,e);if(this[d].has(e)){if(r>this[o])return x(this,this[d].get(e)),!1;var i=this[d].get(e).value;return this[h]&&(this[v]||this[h](e,i.value)),i.now=a,i.maxAge=n,i.value=t,this[s]+=r-i.length,i.length=r,this.get(e),_(this),!0}var u=new w(e,t,r,a,n);return u.length>this[o]?(this[h]&&this[h](e,t),!1):(this[s]+=u.length,this[m].unshift(u),this[d].set(e,this[m].head),_(this),!0)}},{key:"has",value:function(e){if(!this[d].has(e))return!1;var t=this[d].get(e).value;return!y(this,t)}},{key:"get",value:function(e){return b(this,e,!0)}},{key:"peek",value:function(e){return b(this,e,!1)}},{key:"pop",value:function(){var e=this[m].tail;return e?(x(this,e),e.value):null}},{key:"del",value:function(e){x(this,this[d].get(e))}},{key:"load",value:function(e){this.reset();for(var t=Date.now(),n=e.length-1;n>=0;n--){var a=e[n],r=a.e||0;if(0===r)this.set(a.k,a.v);else{var i=r-t;i>0&&this.set(a.k,a.v,i)}}}},{key:"prune",value:function(){var e=this;this[d].forEach((function(t,n){return b(e,n,!1)}))}},{key:"max",set:function(e){if("number"!=typeof e||e<0)throw new TypeError("max must be a non-negative number");this[o]=e||1/0,_(this)},get:function(){return this[o]}},{key:"allowStale",set:function(e){this[u]=!!e},get:function(){return this[u]}},{key:"maxAge",set:function(e){if("number"!=typeof e)throw new TypeError("maxAge must be a non-negative number");this[c]=e,_(this)},get:function(){return this[c]}},{key:"lengthCalculator",set:function(e){var t=this;"function"!=typeof e&&(e=p),e!==this[l]&&(this[l]=e,this[s]=0,this[m].forEach((function(e){e.length=t[l](e.value,e.key),t[s]+=e.length}))),_(this)},get:function(){return this[l]}},{key:"length",get:function(){return this[s]}},{key:"itemCount",get:function(){return this[m].length}}])&&r(t.prototype,n),g&&r(t,g),e}(),b=function(e,t,n){var a=e[d].get(t);if(a){var r=a.value;if(y(e,r)){if(x(e,a),!e[u])return}else n&&(e[f]&&(a.value.now=Date.now()),e[m].unshiftNode(a));return r.value}},y=function(e,t){if(!t||!t.maxAge&&!e[c])return!1;var n=Date.now()-t.now;return t.maxAge?n>t.maxAge:e[c]&&n>e[c]},_=function(e){if(e[s]>e[o])for(var t=e[m].tail;e[s]>e[o]&&null!==t;){var n=t.prev;x(e,t),t=n}},x=function(e,t){if(t){var n=t.value;e[h]&&e[h](n.key,n.value),e[s]-=n.length,e[d].delete(n.key),e[m].removeNode(t)}},w=function e(t,n,r,i,o){a(this,e),this.key=t,this.value=n,this.length=r,this.now=i,this.maxAge=o||0},T=function(e,t,n,a){var r=n.value;y(e,r)&&(x(e,n),e[u]||(r=void 0)),r&&t.call(a,r.value,r.key,e)};e.exports=g},XPeR:function(e,t,n){"use strict";function a(e){var t=this;if(t instanceof a||(t=new a),t.tail=null,t.head=null,t.length=0,e&&"function"==typeof e.forEach)e.forEach((function(e){t.push(e)}));else if(arguments.length>0)for(var n=0,r=arguments.length;n<r;n++)t.push(arguments[n]);return t}function r(e,t,n){var a=t===e.head?new s(n,null,t,e):new s(n,t,t.next,e);return null===a.next&&(e.tail=a),null===a.prev&&(e.head=a),e.length++,a}function i(e,t){e.tail=new s(t,e.tail,null,e),e.head||(e.head=e.tail),e.length++}function o(e,t){e.head=new s(t,null,e.head,e),e.tail||(e.tail=e.head),e.length++}function s(e,t,n,a){if(!(this instanceof s))return new s(e,t,n,a);this.list=a,this.value=e,t?(t.next=this,this.prev=t):this.prev=null,n?(n.prev=this,this.next=n):this.next=null}e.exports=a,a.Node=s,a.create=a,a.prototype.removeNode=function(e){if(e.list!==this)throw new Error("removing node which does not belong to this list");var t=e.next,n=e.prev;return t&&(t.prev=n),n&&(n.next=t),e===this.head&&(this.head=t),e===this.tail&&(this.tail=n),e.list.length--,e.next=null,e.prev=null,e.list=null,t},a.prototype.unshiftNode=function(e){if(e!==this.head){e.list&&e.list.removeNode(e);var t=this.head;e.list=this,e.next=t,t&&(t.prev=e),this.head=e,this.tail||(this.tail=e),this.length++}},a.prototype.pushNode=function(e){if(e!==this.tail){e.list&&e.list.removeNode(e);var t=this.tail;e.list=this,e.prev=t,t&&(t.next=e),this.tail=e,this.head||(this.head=e),this.length++}},a.prototype.push=function(){for(var e=0,t=arguments.length;e<t;e++)i(this,arguments[e]);return this.length},a.prototype.unshift=function(){for(var e=0,t=arguments.length;e<t;e++)o(this,arguments[e]);return this.length},a.prototype.pop=function(){if(this.tail){var e=this.tail.value;return this.tail=this.tail.prev,this.tail?this.tail.next=null:this.head=null,this.length--,e}},a.prototype.shift=function(){if(this.head){var e=this.head.value;return this.head=this.head.next,this.head?this.head.prev=null:this.tail=null,this.length--,e}},a.prototype.forEach=function(e,t){t=t||this;for(var n=this.head,a=0;null!==n;a++)e.call(t,n.value,a,this),n=n.next},a.prototype.forEachReverse=function(e,t){t=t||this;for(var n=this.tail,a=this.length-1;null!==n;a--)e.call(t,n.value,a,this),n=n.prev},a.prototype.get=function(e){for(var t=0,n=this.head;null!==n&&t<e;t++)n=n.next;if(t===e&&null!==n)return n.value},a.prototype.getReverse=function(e){for(var t=0,n=this.tail;null!==n&&t<e;t++)n=n.prev;if(t===e&&null!==n)return n.value},a.prototype.map=function(e,t){t=t||this;for(var n=new a,r=this.head;null!==r;)n.push(e.call(t,r.value,this)),r=r.next;return n},a.prototype.mapReverse=function(e,t){t=t||this;for(var n=new a,r=this.tail;null!==r;)n.push(e.call(t,r.value,this)),r=r.prev;return n},a.prototype.reduce=function(e,t){var n,a=this.head;if(arguments.length>1)n=t;else{if(!this.head)throw new TypeError("Reduce of empty list with no initial value");a=this.head.next,n=this.head.value}for(var r=0;null!==a;r++)n=e(n,a.value,r),a=a.next;return n},a.prototype.reduceReverse=function(e,t){var n,a=this.tail;if(arguments.length>1)n=t;else{if(!this.tail)throw new TypeError("Reduce of empty list with no initial value");a=this.tail.prev,n=this.tail.value}for(var r=this.length-1;null!==a;r--)n=e(n,a.value,r),a=a.prev;return n},a.prototype.toArray=function(){for(var e=new Array(this.length),t=0,n=this.head;null!==n;t++)e[t]=n.value,n=n.next;return e},a.prototype.toArrayReverse=function(){for(var e=new Array(this.length),t=0,n=this.tail;null!==n;t++)e[t]=n.value,n=n.prev;return e},a.prototype.slice=function(e,t){(t=t||this.length)<0&&(t+=this.length),(e=e||0)<0&&(e+=this.length);var n=new a;if(t<e||t<0)return n;e<0&&(e=0),t>this.length&&(t=this.length);for(var r=0,i=this.head;null!==i&&r<e;r++)i=i.next;for(;null!==i&&r<t;r++,i=i.next)n.push(i.value);return n},a.prototype.sliceReverse=function(e,t){(t=t||this.length)<0&&(t+=this.length),(e=e||0)<0&&(e+=this.length);var n=new a;if(t<e||t<0)return n;e<0&&(e=0),t>this.length&&(t=this.length);for(var r=this.length,i=this.tail;null!==i&&r>t;r--)i=i.prev;for(;null!==i&&r>e;r--,i=i.prev)n.push(i.value);return n},a.prototype.splice=function(e,t){e>this.length&&(e=this.length-1),e<0&&(e=this.length+e);for(var n=0,a=this.head;null!==a&&n<e;n++)a=a.next;var i=[];for(n=0;a&&n<t;n++)i.push(a.value),a=this.removeNode(a);null===a&&(a=this.tail),a!==this.head&&a!==this.tail&&(a=a.prev);for(n=2;n<arguments.length;n++)a=r(this,a,arguments[n]);return i},a.prototype.reverse=function(){for(var e=this.head,t=this.tail,n=e;null!==n;n=n.prev){var a=n.prev;n.prev=n.next,n.next=a}return this.head=t,this.tail=e,this};try{n("HwNo")(a)}catch(e){}},ceQ3:function(e,t,n){"use strict";n.d(t,"e",(function(){return s})),n.d(t,"f",(function(){return l})),n.d(t,"g",(function(){return u})),n.d(t,"c",(function(){return c})),n.d(t,"d",(function(){return h})),n.d(t,"a",(function(){return v})),n.d(t,"b",(function(){return m}));var a=n("eqT+");function r(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){if(null==e)return{};var n,a,r=function(e,t){if(null==e)return{};var n,a,r={},i=Object.keys(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||(r[n]=e[n]);return r}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var s=function(e){for(var t=[],n=new RegExp("_bucket($|:)"),a=0;a<e.length;a++){var r=e[a];n.test(r)&&-1===t.indexOf(r)&&t.push(r)}return{values:{__name__:t}}};function l(e){var t=arguments.length>1&&void 0!==arguments[1]&&arguments[1],n={};return e.forEach((function(e){var a=e.__name__,r=o(e,["__name__"]);t&&(n.__name__=n.__name__||[],n.__name__.includes(a)||n.__name__.push(a)),Object.keys(r).forEach((function(e){n[e]||(n[e]=[]),n[e].includes(r[e])||n[e].push(r[e])}))})),{values:n,keys:Object.keys(n)}}var u=/\{[^}]*?\}/,c=/\b(\w+)(!?=~?)("[^"\n]*?")/g;function h(e){var t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:1;if(!e.match(u)){if(e.match(/^[A-Za-z:][\w:]*$/))return{selector:'{__name__="'.concat(e,'"}'),labelKeys:["__name__"]};throw new Error("Query must contain a selector: "+e)}var n=e.slice(0,t),a=n.lastIndexOf("{"),r=n.lastIndexOf("}");if(-1===a)throw new Error("Not inside selector, missing open brace: "+n);if(r>-1&&r>a)throw new Error("Not inside selector, previous selector already closed: "+n);var i=e.slice(t),o=i.indexOf("}"),s=o+t,l=i.indexOf("{"),h=l+t;if(-1===s)throw new Error("Not inside selector, missing closing brace in suffix: "+i);if(l>-1&&h<s)throw new Error("Not inside selector, next selector opens before this one closed: "+i);var v=e.slice(a,s),m={};v.replace(c,(function(n,a,r,i){var o=e.indexOf(n),s=o+a.length+r.length+1,l=o+a.length+r.length+i.length-1;return(t<s||t>l)&&(m[a]={value:i,operator:r}),""}));var d=e.slice(0,a),f=d.match(/[A-Za-z:][\w:]*$/);f&&(m.__name__={value:'"'.concat(f[0],'"'),operator:"="});var p=Object.keys(m).sort(),g=p.map((function(e){return"".concat(e).concat(m[e].operator).concat(m[e].value)})).join(","),b=["{",g,"}"].join("");return{labelKeys:p,selector:b}}function v(e,t){var n=Object.keys(t),r=new RegExp("(\\s|^)(".concat(n.join("|"),")(\\s|$|\\(|\\[|\\{)"),"ig"),i=e.replace(r,(function(e,n,a,r){return"".concat(n).concat(t[a]).concat(r)})).split(/(\+|\-|\*|\/|\%|\^)/),o=/(\)\{|\}\{|\]\{)/;return i.map((function(e){var t=e;return t.match(o)&&(t=function(e,t){var n=e.match(t).index,r=e.substr(0,n+1),i=e.substr(n+1),o=[];i.replace(c,(function(e,t,n,a){return o.push({key:t,operator:n,value:a}),""}));var s=r;return o.filter(Boolean).forEach((function(e){var t=e.value.substr(1,e.value.length-2);s=Object(a.a)(s,e.key,t,e.operator)})),s}(t,o)),t})).join("")}function m(e){if(!e)return e;var t={};for(var n in e){var a=e[n][0];"summary"===a.type&&(t["".concat(n,"_count")]=[{type:"counter",help:"Count of events that have been observed for the base metric (".concat(a.help,")")}],t["".concat(n,"_sum")]=[{type:"counter",help:"Total sum of all observed values for the base metric (".concat(a.help,")")}])}return function(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?r(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):r(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}({},e,{},t)}},"eqT+":function(e,t,n){"use strict";n.d(t,"a",(function(){return u}));var a=n("LvDl"),r=n.n(a),i="by|without|on|ignoring|group_left|group_right|bool|or|and|unless",o=[i,"count|count_values|min|max|avg|sum|stddev|stdvar|bottomk|topk|quantile","true|false|null|__name__|job","abs|absent|ceil|changes|clamp_max|clamp_min|count_scalar|day_of_month|day_of_week|days_in_month|delta|deriv","drop_common_labels|exp|floor|histogram_quantile|holt_winters|hour|idelta|increase|irate|label_replace|ln|log2","log10|minute|month|predict_linear|rate|resets|round|scalar|sort|sort_desc|sqrt|time|vector|year|avg_over_time","min_over_time|max_over_time|sum_over_time|count_over_time|quantile_over_time|stddev_over_time|stdvar_over_time"].join("|").split("|"),s=/([A-Za-z:][\w:]*)\b(?![\(\]{=!",])/g,l=/{([^{]*)}/g;function u(e,t,n,a,r){if(!t||!n)throw new Error("Need label to add to query.");var u;e=e.replace(s,(function(t,n,a){var s,l,c,h,v,m,d=(l=a,c="{",h="}",v=(s=e).slice(l).indexOf(c),(m=s.slice(l).indexOf(h))>-1&&(-1===v||v>m)),f=u&&i.split("|").indexOf(u)>-1,p=n.endsWith(":");return u=n,r||d||p||f||-1!==o.indexOf(n)?n:"".concat(n,"{}")}));for(var c=l.exec(e),v=[],m=0,d="";c;){var f=e.slice(m,c.index),p=h(c[1],t,n,a);m=c.index+c[1].length+2,d=e.slice(c.index+c[0].length),v.push(f,p),c=l.exec(e)}return v.push(d),v.join("")}var c=/(\w+)\s*(=|!=|=~|!~)\s*("[^"]*")/g;function h(e,t,n,a){var i=[];if(e)for(var o=c.exec(e);o;)i.push({key:o[1],operator:o[2],value:o[3]}),o=c.exec(e);var s=a||"=";i.push({key:t,operator:s,value:'"'.concat(n,'"')});var l=r.a.chain(i).uniqWith(r.a.isEqual).compact().sortBy("key").map((function(e){var t=e.key,n=e.operator,a=e.value;return"".concat(t).concat(n).concat(a)})).value().join(",");return"{".concat(l,"}")}t.b=u},itod:function(e,t,n){"use strict";n.d(t,"b",(function(){return a})),n.d(t,"a",(function(){return r}));var a=[{label:"$__interval",sortText:"$__interval"},{label:"1m",sortText:"00:01:00"},{label:"5m",sortText:"00:05:00"},{label:"10m",sortText:"00:10:00"},{label:"30m",sortText:"00:30:00"},{label:"1h",sortText:"01:00:00"},{label:"1d",sortText:"24:00:00"}],r=[].concat([{label:"sum",insertText:"sum",documentation:"Calculate sum over dimensions"},{label:"min",insertText:"min",documentation:"Select minimum over dimensions"},{label:"max",insertText:"max",documentation:"Select maximum over dimensions"},{label:"avg",insertText:"avg",documentation:"Calculate the average over dimensions"},{label:"stddev",insertText:"stddev",documentation:"Calculate population standard deviation over dimensions"},{label:"stdvar",insertText:"stdvar",documentation:"Calculate population standard variance over dimensions"},{label:"count",insertText:"count",documentation:"Count number of elements in the vector"},{label:"count_values",insertText:"count_values",documentation:"Count number of elements with the same value"},{label:"bottomk",insertText:"bottomk",documentation:"Smallest k elements by sample value"},{label:"topk",insertText:"topk",documentation:"Largest k elements by sample value"},{label:"quantile",insertText:"quantile",documentation:"Calculate φ-quantile (0 ≤ φ ≤ 1) over dimensions"}],[{insertText:"abs",label:"abs",detail:"abs(v instant-vector)",documentation:"Returns the input vector with all sample values converted to their absolute value."},{insertText:"absent",label:"absent",detail:"absent(v instant-vector)",documentation:"Returns an empty vector if the vector passed to it has any elements and a 1-element vector with the value 1 if the vector passed to it has no elements. This is useful for alerting on when no time series exist for a given metric name and label combination."},{insertText:"ceil",label:"ceil",detail:"ceil(v instant-vector)",documentation:"Rounds the sample values of all elements in `v` up to the nearest integer."},{insertText:"changes",label:"changes",detail:"changes(v range-vector)",documentation:"For each input time series, `changes(v range-vector)` returns the number of times its value has changed within the provided time range as an instant vector."},{insertText:"clamp_max",label:"clamp_max",detail:"clamp_max(v instant-vector, max scalar)",documentation:"Clamps the sample values of all elements in `v` to have an upper limit of `max`."},{insertText:"clamp_min",label:"clamp_min",detail:"clamp_min(v instant-vector, min scalar)",documentation:"Clamps the sample values of all elements in `v` to have a lower limit of `min`."},{insertText:"count_scalar",label:"count_scalar",detail:"count_scalar(v instant-vector)",documentation:"Returns the number of elements in a time series vector as a scalar. This is in contrast to the `count()` aggregation operator, which always returns a vector (an empty one if the input vector is empty) and allows grouping by labels via a `by` clause."},{insertText:"day_of_month",label:"day_of_month",detail:"day_of_month(v=vector(time()) instant-vector)",documentation:"Returns the day of the month for each of the given times in UTC. Returned values are from 1 to 31."},{insertText:"day_of_week",label:"day_of_week",detail:"day_of_week(v=vector(time()) instant-vector)",documentation:"Returns the day of the week for each of the given times in UTC. Returned values are from 0 to 6, where 0 means Sunday etc."},{insertText:"days_in_month",label:"days_in_month",detail:"days_in_month(v=vector(time()) instant-vector)",documentation:"Returns number of days in the month for each of the given times in UTC. Returned values are from 28 to 31."},{insertText:"delta",label:"delta",detail:"delta(v range-vector)",documentation:"Calculates the difference between the first and last value of each time series element in a range vector `v`, returning an instant vector with the given deltas and equivalent labels. The delta is extrapolated to cover the full time range as specified in the range vector selector, so that it is possible to get a non-integer result even if the sample values are all integers."},{insertText:"deriv",label:"deriv",detail:"deriv(v range-vector)",documentation:"Calculates the per-second derivative of the time series in a range vector `v`, using simple linear regression."},{insertText:"drop_common_labels",label:"drop_common_labels",detail:"drop_common_labels(instant-vector)",documentation:"Drops all labels that have the same name and value across all series in the input vector."},{insertText:"exp",label:"exp",detail:"exp(v instant-vector)",documentation:"Calculates the exponential function for all elements in `v`.\nSpecial cases are:\n* `Exp(+Inf) = +Inf` \n* `Exp(NaN) = NaN`"},{insertText:"floor",label:"floor",detail:"floor(v instant-vector)",documentation:"Rounds the sample values of all elements in `v` down to the nearest integer."},{insertText:"histogram_quantile",label:"histogram_quantile",detail:"histogram_quantile(φ float, b instant-vector)",documentation:"Calculates the φ-quantile (0 ≤ φ ≤ 1) from the buckets `b` of a histogram. The samples in `b` are the counts of observations in each bucket. Each sample must have a label `le` where the label value denotes the inclusive upper bound of the bucket. (Samples without such a label are silently ignored.) The histogram metric type automatically provides time series with the `_bucket` suffix and the appropriate labels."},{insertText:"holt_winters",label:"holt_winters",detail:"holt_winters(v range-vector, sf scalar, tf scalar)",documentation:"Produces a smoothed value for time series based on the range in `v`. The lower the smoothing factor `sf`, the more importance is given to old data. The higher the trend factor `tf`, the more trends in the data is considered. Both `sf` and `tf` must be between 0 and 1."},{insertText:"hour",label:"hour",detail:"hour(v=vector(time()) instant-vector)",documentation:"Returns the hour of the day for each of the given times in UTC. Returned values are from 0 to 23."},{insertText:"idelta",label:"idelta",detail:"idelta(v range-vector)",documentation:"Calculates the difference between the last two samples in the range vector `v`, returning an instant vector with the given deltas and equivalent labels."},{insertText:"increase",label:"increase",detail:"increase(v range-vector)",documentation:"Calculates the increase in the time series in the range vector. Breaks in monotonicity (such as counter resets due to target restarts) are automatically adjusted for. The increase is extrapolated to cover the full time range as specified in the range vector selector, so that it is possible to get a non-integer result even if a counter increases only by integer increments."},{insertText:"irate",label:"irate",detail:"irate(v range-vector)",documentation:"Calculates the per-second instant rate of increase of the time series in the range vector. This is based on the last two data points. Breaks in monotonicity (such as counter resets due to target restarts) are automatically adjusted for."},{insertText:"label_replace",label:"label_replace",detail:"label_replace(v instant-vector, dst_label string, replacement string, src_label string, regex string)",documentation:"For each timeseries in `v`, `label_replace(v instant-vector, dst_label string, replacement string, src_label string, regex string)`  matches the regular expression `regex` against the label `src_label`.  If it matches, then the timeseries is returned with the label `dst_label` replaced by the expansion of `replacement`. `$1` is replaced with the first matching subgroup, `$2` with the second etc. If the regular expression doesn't match then the timeseries is returned unchanged."},{insertText:"ln",label:"ln",detail:"ln(v instant-vector)",documentation:"calculates the natural logarithm for all elements in `v`.\nSpecial cases are:\n * `ln(+Inf) = +Inf`\n * `ln(0) = -Inf`\n * `ln(x < 0) = NaN`\n * `ln(NaN) = NaN`"},{insertText:"log2",label:"log2",detail:"log2(v instant-vector)",documentation:"Calculates the binary logarithm for all elements in `v`. The special cases are equivalent to those in `ln`."},{insertText:"log10",label:"log10",detail:"log10(v instant-vector)",documentation:"Calculates the decimal logarithm for all elements in `v`. The special cases are equivalent to those in `ln`."},{insertText:"minute",label:"minute",detail:"minute(v=vector(time()) instant-vector)",documentation:"Returns the minute of the hour for each of the given times in UTC. Returned values are from 0 to 59."},{insertText:"month",label:"month",detail:"month(v=vector(time()) instant-vector)",documentation:"Returns the month of the year for each of the given times in UTC. Returned values are from 1 to 12, where 1 means January etc."},{insertText:"predict_linear",label:"predict_linear",detail:"predict_linear(v range-vector, t scalar)",documentation:"Predicts the value of time series `t` seconds from now, based on the range vector `v`, using simple linear regression."},{insertText:"rate",label:"rate",detail:"rate(v range-vector)",documentation:"Calculates the per-second average rate of increase of the time series in the range vector. Breaks in monotonicity (such as counter resets due to target restarts) are automatically adjusted for. Also, the calculation extrapolates to the ends of the time range, allowing for missed scrapes or imperfect alignment of scrape cycles with the range's time period."},{insertText:"resets",label:"resets",detail:"resets(v range-vector)",documentation:"For each input time series, `resets(v range-vector)` returns the number of counter resets within the provided time range as an instant vector. Any decrease in the value between two consecutive samples is interpreted as a counter reset."},{insertText:"round",label:"round",detail:"round(v instant-vector, to_nearest=1 scalar)",documentation:"Rounds the sample values of all elements in `v` to the nearest integer. Ties are resolved by rounding up. The optional `to_nearest` argument allows specifying the nearest multiple to which the sample values should be rounded. This multiple may also be a fraction."},{insertText:"scalar",label:"scalar",detail:"scalar(v instant-vector)",documentation:"Given a single-element input vector, `scalar(v instant-vector)` returns the sample value of that single element as a scalar. If the input vector does not have exactly one element, `scalar` will return `NaN`."},{insertText:"sort",label:"sort",detail:"sort(v instant-vector)",documentation:"Returns vector elements sorted by their sample values, in ascending order."},{insertText:"sort_desc",label:"sort_desc",detail:"sort_desc(v instant-vector)",documentation:"Returns vector elements sorted by their sample values, in descending order."},{insertText:"sqrt",label:"sqrt",detail:"sqrt(v instant-vector)",documentation:"Calculates the square root of all elements in `v`."},{insertText:"time",label:"time",detail:"time()",documentation:"Returns the number of seconds since January 1, 1970 UTC. Note that this does not actually return the current time, but the time at which the expression is to be evaluated."},{insertText:"vector",label:"vector",detail:"vector(s scalar)",documentation:"Returns the scalar `s` as a vector with no labels."},{insertText:"year",label:"year",detail:"year(v=vector(time()) instant-vector)",documentation:"Returns the year for each of the given times in UTC."},{insertText:"avg_over_time",label:"avg_over_time",detail:"avg_over_time(range-vector)",documentation:"The average value of all points in the specified interval."},{insertText:"min_over_time",label:"min_over_time",detail:"min_over_time(range-vector)",documentation:"The minimum value of all points in the specified interval."},{insertText:"max_over_time",label:"max_over_time",detail:"max_over_time(range-vector)",documentation:"The maximum value of all points in the specified interval."},{insertText:"sum_over_time",label:"sum_over_time",detail:"sum_over_time(range-vector)",documentation:"The sum of all values in the specified interval."},{insertText:"count_over_time",label:"count_over_time",detail:"count_over_time(range-vector)",documentation:"The count of all values in the specified interval."},{insertText:"quantile_over_time",label:"quantile_over_time",detail:"quantile_over_time(scalar, range-vector)",documentation:"The φ-quantile (0 ≤ φ ≤ 1) of the values in the specified interval."},{insertText:"stddev_over_time",label:"stddev_over_time",detail:"stddev_over_time(range-vector)",documentation:"The population standard deviation of the values in the specified interval."},{insertText:"stdvar_over_time",label:"stdvar_over_time",detail:"stdvar_over_time(range-vector)",documentation:"The population standard variance of the values in the specified interval."}]),i={comment:{pattern:/#.*/},"context-aggregation":{pattern:/((by|without)\s*)\([^)]*\)/,lookbehind:!0,inside:{"label-key":{pattern:/[^(),\s][^,)]*[^),\s]*/,alias:"attr-name"},punctuation:/[()]/}},"context-labels":{pattern:/\{[^}]*(?=})/,greedy:!0,inside:{comment:{pattern:/#.*/},"label-key":{pattern:/[a-z_]\w*(?=\s*(=|!=|=~|!~))/,alias:"attr-name",greedy:!0},"label-value":{pattern:/"(?:\\.|[^\\"])*"/,greedy:!0,alias:"attr-value"},punctuation:/[{]/}},function:new RegExp("\\b(?:".concat(r.map((function(e){return e.label})).join("|"),")(?=\\s*\\()"),"i"),"context-range":[{pattern:/\[[^\]]*(?=])/,inside:{"range-duration":{pattern:/\b\d+[smhdwy]\b/i,alias:"number"}}},{pattern:/(offset\s+)\w+/,lookbehind:!0,inside:{"range-duration":{pattern:/\b\d+[smhdwy]\b/i,alias:"number"}}}],number:/\b-?\d+((\.\d*)?([eE][+-]?\d+)?)?\b/,operator:new RegExp("/[-+*/=%^~]|&&?|\\|?\\||!=?|<(?:=>?|<|>)?|>[>=]?|\\b(?:".concat(["by","group_left","group_right","ignoring","on","offset","without"].join("|"),")\\b"),"i"),punctuation:/[{};()`,.]/};t.c=i}}]);
//# sourceMappingURL=default~lokiPlugin~prometheusPlugin.c8856b8b39626543db12.js.map