/*!
 * AEGIS Client Tracking SDK  v1.0.0
 * Collects login, GPS, device fingerprint, and app-unlock events
 * and streams them to the AEGIS Threat Detection API.
 *
 * Usage:
 *   <script src="aegis.js"></script>
 *   <script>
 *     Aegis.init({ apiUrl: 'https://your-api.com', userId: 'user-123', token: 'Bearer ...' });
 *     Aegis.trackLogin({ success: true, method: 'password' });
 *   </script>
 */

(function (root, factory) {
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = factory();               // CommonJS / Node
  } else if (typeof define === 'function' && define.amd) {
    define([], factory);                       // AMD
  } else {
    root.Aegis = factory();                    // Browser global
  }
}(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  /* ─── State ──────────────────────────────────────────── */
  var _cfg = {
    apiUrl:        'http://localhost:8000',
    apiKey:        null,      // tenant API key — sent as X-Api-Key on every request
    userId:        null,
    token:         null,
    autoFingerprint: true,
    autoGPS:         false,   // must be opt-in — requires permissions
    debug:           false,
    onRisk:          null,    // callback(result) when a score comes back
    batchSize:       5,       // flush queue after N events
    flushInterval:   10000,   // ms — periodic flush
  };

  var _queue   = [];
  var _flushTimer = null;
  var _gpsWatch   = null;
  var _lastGPS    = null;
  var _deviceId   = null;

  /* ─── Logging ────────────────────────────────────────── */
  function log() {
    if (_cfg.debug) {
      console.log('[AEGIS]', Array.prototype.join.call(arguments, ' '));
    }
  }

  /* ─── HTTP helper ────────────────────────────────────── */
  function post(path, body) {
    var url = _cfg.apiUrl + path;
    var headers = { 'Content-Type': 'application/json' };
    if (_cfg.token)  headers['Authorization'] = _cfg.token.startsWith('Bearer ') ? _cfg.token : 'Bearer ' + _cfg.token;
    if (_cfg.apiKey) headers['X-Api-Key'] = _cfg.apiKey;

    if (typeof fetch !== 'undefined') {
      return fetch(url, { method: 'POST', headers: headers, body: JSON.stringify(body) })
        .then(function (r) { return r.ok ? r.json() : Promise.reject(r.status); });
    }
    // XHR fallback
    return new Promise(function (resolve, reject) {
      var xhr = new XMLHttpRequest();
      xhr.open('POST', url, true);
      Object.keys(headers).forEach(function (k) { xhr.setRequestHeader(k, headers[k]); });
      xhr.onload  = function () { resolve(JSON.parse(xhr.responseText)); };
      xhr.onerror = function () { reject(xhr.status); };
      xhr.send(JSON.stringify(body));
    });
  }

  /* ─── SHA-1 (for HIBP k-anonymity) ──────────────────── */
  function sha1Async(str) {
    if (typeof crypto !== 'undefined' && crypto.subtle) {
      var buf = new TextEncoder().encode(str);
      return crypto.subtle.digest('SHA-1', buf).then(function (ab) {
        return Array.from(new Uint8Array(ab)).map(function (b) {
          return b.toString(16).padStart(2, '0');
        }).join('').toUpperCase();
      });
    }
    return Promise.resolve(null);
  }

  /* ─── Device Fingerprint ─────────────────────────────── */
  function collectFingerprint() {
    var nav  = (typeof navigator !== 'undefined') ? navigator : {};
    var scr  = (typeof screen    !== 'undefined') ? screen    : {};
    var fp   = {
      user_agent:           nav.userAgent        || '',
      platform:             nav.platform         || '',
      language:             nav.language         || '',
      hardware_concurrency: nav.hardwareConcurrency || 0,
      color_depth:          scr.colorDepth       || 0,
      screen_resolution:    (scr.width || 0) + 'x' + (scr.height || 0),
      timezone:             (typeof Intl !== 'undefined') ? Intl.DateTimeFormat().resolvedOptions().timeZone : '',
      touch_support:        (typeof nav.maxTouchPoints !== 'undefined') ? nav.maxTouchPoints > 0 : false,
    };
    // WebGL renderer
    try {
      var canvas = document.createElement('canvas');
      var gl     = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      if (gl) {
        var dbg = gl.getExtension('WEBGL_debug_renderer_info');
        if (dbg) fp.webgl_renderer = gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL);
      }
    } catch (e) {}

    // Stable device ID from fingerprint hash
    var raw = JSON.stringify(fp);
    var hash = 0;
    for (var i = 0; i < raw.length; i++) { hash = ((hash << 5) - hash) + raw.charCodeAt(i); hash |= 0; }
    _deviceId = Math.abs(hash).toString(16).padStart(8, '0');
    fp.device_id = _deviceId;
    return fp;
  }

  /* ─── GPS Tracking ───────────────────────────────────── */
  function startGPSTracking() {
    if (typeof navigator === 'undefined' || !navigator.geolocation) {
      log('GPS not available');
      return;
    }
    _gpsWatch = navigator.geolocation.watchPosition(
      function (pos) {
        _lastGPS = { lat: pos.coords.latitude, lng: pos.coords.longitude,
                     timestamp: pos.timestamp, speed: pos.coords.speed,
                     accuracy: pos.coords.accuracy };
        log('GPS update', _lastGPS.lat.toFixed(4), _lastGPS.lng.toFixed(4));
      },
      function (err) { log('GPS error', err.message); },
      { enableHighAccuracy: true, maximumAge: 30000 }
    );
  }

  function stopGPSTracking() {
    if (_gpsWatch !== null && typeof navigator !== 'undefined' && navigator.geolocation) {
      navigator.geolocation.clearWatch(_gpsWatch);
      _gpsWatch = null;
    }
  }

  /* ─── Queue & Flush ──────────────────────────────────── */
  function enqueue(eventType, payload) {
    _queue.push({ type: eventType, ts: Date.now(), payload: payload });
    log('queued', eventType, '— queue size:', _queue.length);
    if (_queue.length >= _cfg.batchSize) flush();
  }

  function flush() {
    if (_queue.length === 0) return;
    var batch = _queue.splice(0, _queue.length);
    log('flushing', batch.length, 'events');

    // For each event, POST to the appropriate endpoint
    batch.forEach(function (item) {
      var endpoint = {
        login:     '/risk/unified',
        gps:       '/gps/score',
        device:    '/device/score',
        password:  '/breach/check/password',
        fraud:     '/fraud/score',
        unified:   '/risk/unified',
      }[item.type] || '/risk/unified';

      post(endpoint, Object.assign({ user_id: _cfg.userId }, item.payload))
        .then(function (result) {
          log('result', item.type, JSON.stringify(result).slice(0, 80));
          if (typeof _cfg.onRisk === 'function') {
            _cfg.onRisk({ event: item.type, result: result });
          }
        })
        .catch(function (err) { log('post error', err); });
    });
  }

  /* ─── Public API ─────────────────────────────────────── */
  var Aegis = {

    /**
     * Initialize the SDK.
     * @param {object} opts  Configuration object (see _cfg defaults above).
     */
    init: function (opts) {
      Object.assign(_cfg, opts || {});

      if (_cfg.autoFingerprint) {
        var fp = collectFingerprint();
        post('/device/score', { user_id: _cfg.userId, fingerprint: fp })
          .then(function (r) {
            log('device fingerprint score:', r.risk_score);
            if (typeof _cfg.onRisk === 'function') _cfg.onRisk({ event: 'device', result: r });
          })
          .catch(function () {});
      }

      if (_cfg.autoGPS) startGPSTracking();

      // Periodic flush
      if (_cfg.flushInterval > 0) {
        _flushTimer = setInterval(flush, _cfg.flushInterval);
      }

      log('initialized — user:', _cfg.userId, '— API:', _cfg.apiUrl);
      return this;
    },

    /**
     * Track a login event.
     * @param {object} data  { success, method, mfa_used, hour_of_day, failed_attempts }
     */
    trackLogin: function (data) {
      data = data || {};
      var now  = new Date();
      var payload = {
        user_id:              _cfg.userId,
        login_data: {
          hour_of_day:        data.hour_of_day   !== undefined ? data.hour_of_day   : now.getHours(),
          failed_10min:       data.failed_attempts !== undefined ? data.failed_attempts : 0,
          is_new_comp:        data.new_device      ? 1 : 0,
          comp_deg:           data.comp_deg        || 0,
          user_deg:           0,
          time_since_user_last: data.time_since_last || 3600,
        },
      };
      if (_lastGPS) {
        payload.gps_data = { trajectory: [_lastGPS], spoof_probability: 0 };
      }
      post('/risk/unified', payload)
        .then(function (r) {
          log('login risk:', r.unified_score, r.risk_level);
          if (typeof _cfg.onRisk === 'function') _cfg.onRisk({ event: 'login', result: r });
        })
        .catch(function () { enqueue('login', payload); });
      return this;
    },

    /**
     * Track a password being set/used (breach check).
     * The password is SHA-1 hashed client-side; only the first 5 chars are sent.
     * @param {string} password
     */
    trackPassword: function (password) {
      if (!password) return this;
      var self = this;
      sha1Async(password).then(function (hash) {
        var prefix = hash ? hash.slice(0, 5) : null;
        post('/breach/check/password', {
          password:    password,       // server also hashes — kept for strength scoring
          hash_prefix: prefix,
          user_id:     _cfg.userId,
        })
          .then(function (r) {
            log('password breach:', r.risk_level, '— pwned:', r.is_pwned);
            if (typeof _cfg.onRisk === 'function') _cfg.onRisk({ event: 'password', result: r });
          })
          .catch(function () {});
      });
      return this;
    },

    /**
     * Track a financial transaction.
     * @param {object} tx  { amount, is_international, merchant_id }
     */
    trackTransaction: function (tx) {
      tx = tx || {};
      var now = new Date();
      var payload = {
        amount:             tx.amount         || 0,
        is_international:   tx.is_international ? true : false,
        hour:               tx.hour           !== undefined ? tx.hour : now.getHours(),
        tx_count_1h:        tx.tx_count_1h    || 1,
        time_since_last_tx: tx.time_since_last_tx || 3600,
        amount_ratio:       tx.amount_ratio   || 1.0,
        merchant_freq_user: tx.merchant_freq  || 1,
        user_id:            _cfg.userId,
      };
      post('/fraud/score', payload)
        .then(function (r) {
          log('fraud score:', r.fraud_probability);
          if (typeof _cfg.onRisk === 'function') _cfg.onRisk({ event: 'fraud', result: r });
        })
        .catch(function () { enqueue('fraud', payload); });
      return this;
    },

    /**
     * Track an app-unlock attempt (mobile context — call from native bridge).
     * @param {object} data  { success, biometric, app_bundle, rooted }
     */
    trackAppUnlock: function (data) {
      data = data || {};
      var fp = collectFingerprint();
      var payload = {
        user_id:               _cfg.userId,
        fingerprint:           fp,
        failed_biometric:      !data.success,
        failed_biometric_count: data.failed_count || (data.success ? 0 : 1),
        app_unlock_attempted:  true,
        app_bundle:            data.app_bundle || '',
        unusual_time:          (new Date().getHours() < 5 || new Date().getHours() > 23),
        rooted_jailbroken:     data.rooted || false,
        emulator_detected:     data.emulator || false,
      };
      post('/device/score', payload)
        .then(function (r) {
          log('device score after unlock:', r.risk_score);
          if (typeof _cfg.onRisk === 'function') _cfg.onRisk({ event: 'app_unlock', result: r });
        })
        .catch(function () { enqueue('device', payload); });
      return this;
    },

    /**
     * Send current GPS position for spoofing analysis.
     * @param {number} lat
     * @param {number} lng
     */
    pushGPS: function (lat, lng) {
      var point = { lat: lat, lng: lng, timestamp: Date.now() };
      var trajectory = _lastGPS ? [_lastGPS, point] : [point];
      _lastGPS = point;
      post('/gps/score', { trajectory: trajectory, user_id: _cfg.userId })
        .then(function (r) {
          log('gps score:', r.risk_score);
          if (typeof _cfg.onRisk === 'function') _cfg.onRisk({ event: 'gps', result: r });
        })
        .catch(function () {});
      return this;
    },

    /**
     * Run a full unified risk score with all available signals.
     * Resolves with the full result object.
     */
    scoreNow: function (extraData) {
      var fp      = collectFingerprint();
      var now     = new Date();
      var payload = Object.assign({
        user_id:     _cfg.userId,
        event_id:    'sdk_' + Date.now(),
        login_data:  { hour_of_day: now.getHours(), failed_10min: 0, is_new_comp: 0 },
        fusion_strategy: 'weighted_average',
      }, extraData || {});

      if (_lastGPS) {
        payload.gps_data = { trajectory: [_lastGPS], spoof_probability: 0 };
      }

      return post('/risk/unified', payload).then(function (r) {
        if (typeof _cfg.onRisk === 'function') _cfg.onRisk({ event: 'unified', result: r });
        return r;
      });
    },

    /** Manually flush the event queue. */
    flush: flush,

    /** Destroy the SDK — stop timers and GPS. */
    destroy: function () {
      if (_flushTimer) { clearInterval(_flushTimer); _flushTimer = null; }
      stopGPSTracking();
      flush();
      log('destroyed');
    },

    /** Get the stable device ID (computed from fingerprint). */
    getDeviceId: function () {
      if (!_deviceId) collectFingerprint();
      return _deviceId;
    },

    /** Return current queue length (unflushed events). */
    queueLength: function () { return _queue.length; },
  };

  return Aegis;
}));
