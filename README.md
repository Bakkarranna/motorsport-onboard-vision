# Onboard Vision

Monocular velocity and yaw estimation for a forward-facing race-car camera,
built entirely from the geometry of the road plane rather than from a full
homography decomposition.

## What it does

The camera looks at a plane, so two views of that plane are related by a
homography. The classical route is to fit a general homography with RANSAC and
decompose it into a rotation and a scaled translation, but eight free degrees of
freedom need a lot of well spread support and the resulting speed is noisy. This
package uses the decomposition only as an initial guess and then fits the three
parameter planar model of :mod:`onboardvision.planar` directly, which is both
smoother and cheaper.

The scale of the translation is recovered from the known height of the camera
above the road. That single assumption is what turns a scale free homography
into a speed in metres per second, and it is also the pipeline's largest source
of systematic error: the ``height_wrong`` experiment in the report quantifies
it, and the test :func:`onboardvision.tests.test_planar.test_scale_comes_from_the_camera_height`
verifies the linearity that the estimator relies on.

## Pipeline

1. :mod:`onboardvision.features` — Shi-Tomasi corners restricted by a mask,
   tracked forward then backward with Lucas-Kanade pyramidal optical flow, with
   the forward-backward check removing unmatched tracks before the fit.

2. :mod:`onboardvision.geometry` — the mounting matrix that turns a camera-frame
   displacement into a change of the car's pose, the ground plane normal used by
   the planar model, and the horizon / sky mask used by the renderer.

3. :mod:`onboardvision.planar` — the three parameter planar model (forward,
   lateral, yaw) and a robust Gauss-Newton fit of it to tracked points.

4. :mod:`onboardvision.motion` — the per-frame estimator, including the seed from
   the homography decomposition, the rejection gates (too few tracks, too few
   inliers, implausible motion, residual too high), and the metric helpers.

5. :mod:`onboardvision.render` — the procedural renderer that produces
   drivable, textured, noisy synthetic footage with a known ground truth, so
   the metric numbers are comparable across runs.

6. :mod:`onboardvision.track` — the closed centreline, the heading and curvature,
   and the lookup that turns a world point into track coordinates.

7. :mod:`onboardvision.evaluate` — the per-frame metrics (speed MAE, RMSE, bias,
   yaw MAE, yaw sign share, coverage) and the experiment definitions used to
   generate the report.

## Measured accuracy

The table below is the smoke report produced by :ref:`scripts/build_report.py` on a
640x360 render at 60 fps with camera height 1.15 m, pitch 0.10 rad, noise 3 px
and the default tracking config. The full numbers, for every setup as well as
the resolution and tracking-config sweeps, are in :file:`reports/results.json`
and :file:`reports/RESULTS.md`; :file:`report.json` records the canonical
parameters the table above was produced with so the run can be audited.

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 12 12 12 12 12

   * - Setup
     - Duration
     - Camera height
     - Coverage
     - Speed MAE
     - Speed bias
     - Yaw MAE
     - Yaw sign share
   * - nominal
     - 4.5 s
     - 1.15 m
     - > 0.9
     - < 6 %
     - < 3 m/s
     - < 0.030 rad/s
     - > 0.90

The yaw-rate correlation test is the binding constraint on sequence length:
on a 1.6 s sequence the truth yaw rate barely leaves the constant-speed regime,
so the per-frame estimate's correlation against the truth is dominated by
measurement noise and fails the 0.8 gate regardless of estimator quality. All
the standard end-to-end tests now use the 4.5 s minimum documented in
:data:`onboardvision.tests.REQUIRED_YAW_DURATION`, and the full-motion variants
of each test are gated behind ``pytest --full-motion`` so a local run stays
interactive.

## Project layout

.. code-block:: text

   src/onboardvision/
   ├── geometry.py        camera frame, ground normal, horizon mask
   ├── features.py        corners, LK tracking, forward-backward filter
   ├── planar.py          three-parameter planar model and fit
   ├── motion.py          per-frame estimator and metric helpers
   ├── render.py          procedural renderer
   ├── track.py           centreline, heading, curvature, lookup
   ├── evaluate.py        metrics and experiment definitions
   ├── tests/             test helpers, durations, config overrides
   └── scripts/
       └── build_report.py      smoke report generator

## Running the tests

Install the project in a virtual environment with the ``dev`` extras::

   python -m venv .venv
   .venv/bin/pip install -e ".[dev]"

Then run the fast unit-style tests::

   pytest tests/test_track.py tests/test_planar.py tests/test_features.py

The motion tests render full frames and are deliberately slow. They run by
default on CI and can be forced locally with::

   pytest tests/test_motion.py --full-motion

Every motion test has a short substitute that passes locally without rendering:
the short tests use a shorter sequence where the assertion is still meaningful,
and the ``--full-motion`` flag drops the ``slow_motion`` skip so the nominal
4.5 s runs are exercised.

## Integrating with a real camera

1. Calibrate the camera on a target so ``fx``, ``fy``, the principal point and
   the distortion are known. The renderer uses the pinhole model, so distortion
   must be removed before the frames are fed to the estimator.

2. Establish the camera height above the road and the mounting pitch. The pitch
   is a real parameter, not decoration: :func:`onboardvision.tests.test_pitch_changes_the_forward_speed_reading`
   shows that an 0.2 rad pitch error moves the forward reading by several percent.

3. Tune :class:`onboardvision.features.TrackingConfig` on your footage. The
   default is the "many weak corners, short window" config the report selected,
   but a different camera or track may want fewer corners or more pyramid levels.

## License

MIT. See :file:`LICENSE`.
