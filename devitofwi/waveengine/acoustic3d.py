__all__ = ["AcousticWave3D"]

from typing import Any, Optional, NewType, Type, Tuple

import numpy as np
import matplotlib.pyplot as plt

from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray, SamplingLike
from tqdm.auto import tqdm

from examples.seismic import AcquisitionGeometry, Model
from examples.seismic.model import SeismicModel

from devitofwi.devito.acoustic.wavesolver import AcousticWaveSolver
from devitofwi.nonlinear import NonlinearOperator
from devitofwi.devito.source import CustomSource
from devitofwi.devito.utils import clear_devito_cache

try:
    from mpi4py import MPI
    mpitype = MPI.Comm
except:
    mpitype = Any

MPIType = NewType("MPIType", mpitype)


class AcousticWave3D(NonlinearOperator):
    """Devito Acoustic 3D propagator.

    This class provides functionalities to model acoustic data and 
    perform full-waveform inversion with the Devito Acoustic propagator
    in three dimensions.

    Parameters
    ----------
    shape : :obj:`tuple`
        Model shape ``(nx, ny, nz)``
    origin : :obj:`tuple`
        Model origin in km ``(ox, oy, oz)``
    spacing : :obj:`tuple`
        Model spacing in km ``(dx, dy, dz)``
    src_x : :obj:`numpy.ndarray`
        Source x-coordinates in km
    src_y : :obj:`numpy.ndarray`
        Source y-coordinates in km
    src_z : :obj:`numpy.ndarray` or :obj:`float`
        Source z-coordinates in km
    rec_x : :obj:`numpy.ndarray`
        Receiver x-coordinates in km. Either a 1D array of length ``nrec`` 
        (fixed receivers, shared across shots) or a 2D array of shape 
        ``(nsrc, nrec)`` (per-shot receivers, e.g. streamer acquisition).
    rec_y : :obj:`numpy.ndarray`
        Receiver y-coordinates in km. Same shape conventions as ``rec_x``.
    rec_z : :obj:`numpy.ndarray` or :obj:`float`
        Receiver z-coordinates in km. Scalar, 1D, or 2D as for ``rec_x``.
    t0 : :obj:`float`
        Initial time in s
    tn : :obj:`float`
        Final time in s
    dt : :obj:`float`, optional
        Time step in s (if not provided this is directly inferred by devito)
    vp : :obj:`numpy.ndarray`, optional
        Velocity model in km/s for modelling of size 
        :math:`n_x \\times n_y \\times n_z`
        (use ``None`` if the data is already available)
    vprange : :obj:`tuple`, optional
        Velocity range in km/s ``(vmin, vmax)``, to be used in loss and gradient computations
        (can be provided instead of ``vp`` to create a propagator with a time axis 
        that is consistent with that of the data modelled with ``vp``)
    space_order : :obj:`int`, optional
        Spatial ordering of FD stencil
    nbl : :obj:`int`, optional
        Number ordering of samples in absorbing boundaries
    src_type : :obj:`str`, optional
        Source type
    f0 : :obj:`float`, optional
        Source peak frequency in Hz
    wav : :obj:`numpy.ndarray`, optional
        Wavelet (if provided ``src_type`` will be ignored)
    fs : :obj:'bool', optional
        Use free surface boundary at the top of the model.
    checkpointing : :obj:`bool`, optional
        Use checkpointing (``True``) or not (``False``). Note that
        using checkpointing is needed when dealing with large models.
        Cannot be used with snapshotting (factor).
    factor : :obj:`int`, optional 
        Subsampling factor to use snapshots of the wavefield to compute the gradient.
        Cannot be used with checkpointing.
    loss : :obj:`Type`, optional
        Loss object.
    dtype : :obj:`str`, optional
        Type of elements in input array.
    clearcache : :obj:`bool`, optional
        Clear devito cache (``True``) or not (``False``) after every modelling step
    base_comm : :obj:`mpi4py.MPI.Comm`, optional
        Base MPI Communicator. Defaults to ``mpi4py.MPI.COMM_WORLD``.

    Notes
    -----
    Streamer acquisition (or any acquisition where receivers vary between shots)
    is supported by passing 2D ``rec_x``, ``rec_y``, ``rec_z`` arrays of shape
    ``(nsrc, nrec)``. The number of receivers per shot must be constant.

    The ``sub_gradient`` per-shot subdomain feature available in
    :class:`AcousticWave2D` is not currently supported in 3D and is
    deliberately omitted from this class.

    """

    def __init__(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        src_x: NDArray,
        src_y: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
        rec_y: NDArray,
        rec_z: NDArray,
        t0: float,
        tn: float,
        dt: Optional[float] = None,
        vp: Optional[NDArray] = None,
        vprange: Optional[Tuple] = None,
        space_order: Optional[int] = 4,
        nbl: Optional[int] = 20,
        src_type: Optional[str] = "Ricker",
        f0: Optional[float] = 20.0,
        wav: Optional[NDArray] = None,
        fs: Optional[bool] = False,
        checkpointing: Optional[bool] = False,
        factor: Optional[int] = None,
        loss: Optional[Type] = None,
        dtype: Optional[DTypeLike] = "float32",
        clearcache: Optional[bool] = False,
        base_comm: Optional[MPIType] = None,
    ) -> None:

        # Check to ensure that vp or vprange is provided
        if vp is None and vprange is None:
            raise ValueError("Provide either vp or vprange, not none...")
        elif vp is not None and vprange is not None:
            raise ValueError("Provide either vp or vprange, not both...")

        # Create vp if not provided and vprange is available
        if vprange is not None:
            vp = vprange[0] * np.ones(shape)
            vp[:, :, -1] = vprange[1]

        # Geometry parameters
        self.src = (src_x, src_y, src_z)

        # Normalize receivers to shape (nsrc, nrec). Each input may be a scalar,
        # a 1D array of length nrec (fixed geometry, shared across shots), or
        # a 2D array of shape (nsrc, nrec) (per-shot receivers, e.g. streamer).
        # A scalar is taken to mean "this value at every receiver" — it is
        # promoted to a full 1D array of length nrec, inferred from the other
        # axes — so e.g. rec_z=0.025 paired with 1D rec_x of length 8 means
        # all 8 receivers sit at z=0.025.
        nsrc = np.asarray(src_x).size
        recs = [np.asarray(r) for r in (rec_x, rec_y, rec_z)]
        # Detect per-shot receivers from the raw user input: True if any axis
        # was passed as a 2D (nsrc, nrec) array. This must be done before
        # _normalize_receivers broadcasts 1D inputs, otherwise the flag
        # cannot distinguish "user passed 1D, we broadcast" from "user passed
        # 2D with nsrc rows".
        self.per_shot_recs = any(r.ndim == 2 for r in recs)
        non_scalar = [r for r in recs if r.ndim > 0]
        if not non_scalar:
            raise ValueError("At least one of rec_x, rec_y, rec_z must be array-like")
        nrec_candidates = {r.shape[-1] for r in non_scalar}
        if len(nrec_candidates) > 1:
            raise ValueError(
                f"rec_x, rec_y, rec_z disagree on number of receivers; "
                f"got trailing dimensions {nrec_candidates}"
            )
        nrec = nrec_candidates.pop()
        # Promote scalars to 1D length nrec
        recs = [np.full(nrec, float(r)) if r.ndim == 0 else r for r in recs]
        rec_x = self._normalize_receivers(recs[0], nsrc, "rec_x")
        rec_y = self._normalize_receivers(recs[1], nsrc, "rec_y")
        rec_z = self._normalize_receivers(recs[2], nsrc, "rec_z")
        if not (rec_x.shape == rec_y.shape == rec_z.shape):
            raise ValueError(
                f"rec_x, rec_y, rec_z must have the same shape after broadcasting; "
                f"got {rec_x.shape}, {rec_y.shape}, {rec_z.shape}"
            )
        self.rec = (rec_x, rec_y, rec_z)

        # Modelling parameters
        self.shape = shape
        self.origin = origin
        self.spacing = spacing
        self.t0 = t0
        self.tn = tn
        self.dt = dt
        self.space_order = space_order
        self.nbl = nbl
        self.src_type = src_type
        self.f0 = f0
        self.wav = wav
        self.fs = fs
        self.checkpointing = checkpointing
        self.factor = factor
        self.clearcache = clearcache

        # Store model
        self.vp = vp

        # Inversion parameters
        self.loss = loss
        self.losshistory = []

        # MPI parameters
        self.base_comm = base_comm

        super().__init__(size=np.prod(shape), dtype=dtype)

    @staticmethod
    def _normalize_receivers(rec: Any, nsrc: int, name: str) -> NDArray:
        """Normalize a receiver coordinate array to shape ``(nsrc, nrec)``.

        Accepts a 1D array of length ``nrec`` (fixed geometry, broadcast 
        across shots) or a 2D array of shape ``(nsrc, nrec)`` (per-shot 
        geometry, e.g. streamer acquisition). Returns a 2D array with 
        leading axis equal to ``nsrc``.

        Scalar inputs are handled in :meth:`__init__` and are not 
        accepted here directly.

        Parameters
        ----------
        rec : 1D or 2D array-like
            Receiver coordinate input
        nsrc : :obj:`int`
            Number of sources
        name : :obj:`str`
            Name of the coordinate (used in error messages)

        Returns
        -------
        rec : :obj:`numpy.ndarray`
            Receiver coordinate of shape ``(nsrc, nrec)``

        """
        rec = np.asarray(rec)
        if rec.ndim == 1:
            return np.broadcast_to(rec, (nsrc, rec.size)).copy()
        if rec.ndim == 2:
            if rec.shape[0] != nsrc:
                raise ValueError(
                    f"{name} has leading dimension {rec.shape[0]} but expected "
                    f"{nsrc} (one row per source)"
                )
            return rec
        raise ValueError(f"{name} must be 1D or 2D; got {rec.ndim}D")

    @staticmethod
    def _crop_model(m: NDArray, nbl: int, fs: bool) -> NDArray:
        """Remove absorbing boundaries from model"""
        if fs:
            return m[nbl:-nbl, nbl:-nbl, :-nbl]
        else:
            return m[nbl:-nbl, nbl:-nbl, nbl:-nbl]

    def _create_model(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        vp: NDArray,
        space_order: int = 4,
        nbl: int = 20,
        fs: bool = False,
    ) -> None:
        """Create model

        Parameters
        ----------
        shape : :obj:`numpy.ndarray`
            Model shape ``(nx, ny, nz)``
        origin : :obj:`numpy.ndarray`
            Model origin in km ``(ox, oy, oz)``
        spacing : :obj:`numpy.ndarray`
            Model spacing in km ``(dx, dy, dz)``
        vp : :obj:`numpy.ndarray`
            Velocity model in km/s
        space_order : :obj:`int`, optional
            Spatial ordering of FD stencil
        nbl : :obj:`int`, optional
            Number ordering of samples in absorbing boundaries
        fs : :obj:'bool', optional
            Use free surface boundary at the top of the model.

        Returns
        -------
        model : :obj:`examples.seismic.model.SeismicModel`
            Model

        """
        model = Model(
            space_order=space_order,
            vp=vp,
            origin=origin,
            shape=shape,
            dtype=np.float32,
            spacing=spacing,
            nbl=nbl,
            bcs="damp",
            fs=fs,
        )
        return model

    def _create_geometry(
        self,
        model,
        src_x: NDArray,
        src_y: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
        rec_y: NDArray,
        rec_z: NDArray,
        t0: float,
        tn: float,
        src_type: str,
        f0: float = 20.0,
        dt: float = None
    ) -> None:
        """Create geometry and time axis

        Parameters
        ----------
        model : :obj:`examples.seismic.model.SeismicModel`
            Model
        src_x : :obj:`numpy.ndarray`
            Source x-coordinates in km
        src_y : :obj:`numpy.ndarray`
            Source y-coordinates in km
        src_z : :obj:`numpy.ndarray` or :obj:`float`
            Source z-coordinates in km
        rec_x : :obj:`numpy.ndarray`
            Receiver x-coordinates in km
        rec_y : :obj:`numpy.ndarray`
            Receiver y-coordinates in km
        rec_z : :obj:`numpy.ndarray` or :obj:`float`
            Receiver z-coordinates in km
        t0 : :obj:`float`
            Initial time in s
        tn : :obj:`float`
            Final time in s
        src_type : :obj:`str`
            Source type
        f0 : :obj:`float`, optional
            Source peak frequency in Hz
        dt : :obj:`float`, optional
            Time step time in s (if provided, the geometry time_axis is
            recreated with this time step)

        """
        nsrc, nrec = len(src_x), len(rec_x)
        src_coordinates = np.empty((nsrc, 3))
        src_coordinates[:, 0] = src_x
        src_coordinates[:, 1] = src_y
        src_coordinates[:, 2] = src_z

        rec_coordinates = np.empty((nrec, 3))
        rec_coordinates[:, 0] = rec_x
        rec_coordinates[:, 1] = rec_y
        rec_coordinates[:, 2] = rec_z

        geometry = AcquisitionGeometry(
            model,
            rec_coordinates,
            src_coordinates,
            t0,
            tn,
            src_type=src_type,
            f0=None if f0 is None else f0,
            fs=self.fs,
        )

        # Resample geometry to user defined dt
        if dt is not None:
            geometry.resample(dt)

        return geometry

    def model_and_geometry(self):
        model = self._create_model(self.shape, self.origin, self.spacing,
                                   self.vp, self.space_order, self.nbl, self.fs)
        geometry = self._create_geometry(model,
                                         self.src[0][:1], self.src[1][:1], self.src[2][:1],
                                         self.rec[0][0], self.rec[1][0], self.rec[2][0],
                                         self.t0, self.tn, self.src_type, f0=self.f0, dt=self.dt)
        return model, geometry

    def _mod_oneshot(self, model: SeismicModel, isrc: int, dt: float = None) -> NDArray:
        """FD modelling for one shot

        Parameters
        ----------
        model : :obj:`examples.seismic.model.SeismicModel`
            Model
        isrc : :obj:`int`
            Index of source to model
        dt : :obj:`float`, optional
            Time sampling in s used to resample modelled data

        Returns
        -------
        d : :obj:`np.ndarray`
            Data of size ``nt \\times nr`` (the receiver axis is flat: 
            the receiver grid layout chosen by the user is not preserved)
        dt : :obj:`float`, optional
            Time sampling in s of modelled data

        """
        # Create geometry using receivers for this shot
        geometry = self._create_geometry(model,
                                         self.src[0][:1], self.src[1][:1], self.src[2][:1],
                                         self.rec[0][isrc], self.rec[1][isrc], self.rec[2][isrc],
                                         self.t0, self.tn, self.src_type, f0=self.f0, dt=self.dt)

        # Update source location in geometry
        geometry.src_positions[0, :] = (self.src[0][isrc], self.src[1][isrc], self.src[2][isrc])

        # Re-create source (if wav is not None)
        if self.wav is None:
            src = geometry.src
        else:
            src = CustomSource(name='src', grid=model.grid,
                               wav=self.wav, npoint=1,
                               time_range=geometry.time_axis)
            geometry.src_positions[0, :] = (self.src[0][isrc], self.src[1][isrc], self.src[2][isrc])
            src.coordinates.data[0, :] = (self.src[0][isrc], self.src[1][isrc], self.src[2][isrc])

        # Solve
        solver = AcousticWaveSolver(model, geometry,
                                    space_order=self.space_order)
        d, _, _, _ = solver.forward(vp=model.vp, src=src, autotune=True)

        # Resample
        if dt is None:
            dt = geometry.dt
            d = d.data.copy()
        else:
            d = d.resample(dt).data.copy()

        return d, dt

    def mod_allshots(self, dt=None, show_progress=True) -> NDArray:
        """FD modelling for all shots

        Parameters
        ----------
        dt : :obj:`float`, optional
            Time sampling used to resample modelled data in s
        show_progress : :obj:`bool`, optional
            Display a tqdm progress bar

        Returns
        -------
        dtot : :obj:`np.ndarray`
            Data for all shots of size ``nsrc \\times nt \\times nr``
        dt : :obj:`float`, optional
            Time sampling in s of modelled data

        """
        # Create model
        model = self._create_model(
            self.shape, self.origin, self.spacing,
            self.vp, self.space_order, self.nbl, self.fs
        )

        # Run modelling
        nsrc = self.src[0].size
        dtot = []

        shot_iterator = range(nsrc)
        if show_progress:
            shot_iterator = tqdm(shot_iterator, desc="Modelling shots")

        for isrc in shot_iterator:
            d, dt = self._mod_oneshot(model, isrc, dt)
            dtot.append(d)

            if self.clearcache:
                clear_devito_cache()

        dtot = np.array(dtot).reshape(nsrc, d.shape[0], d.shape[1])

        return dtot, dt

    def mod_allshots_mpi(self, dt=None) -> NDArray:
        """FD modelling for all shots with mpi gathering

        Parameters
        ----------
        dt : :obj:`float`, optional
            Time sampling used to resample modelled data in s

        Returns
        -------
        dtot : :obj:`np.ndarray`
            Data for all shots
        dt : :obj:`float`, optional
            Time sampling in s of modelled data

        """
        rank = self.base_comm.Get_rank()

        dtotrank, dt = self.mod_allshots(
            dt=dt,
            show_progress=(rank == 0)
        )

        # gather shots from all ranks
        dtot = np.concatenate(self.base_comm.allgather(dtotrank), axis=0)

        return dtot, dt

    def _adjoint_source(self, d_syn, isrc):
        """Adjoint source computation

        Note to self, takes flatten inputs and returns flatten outputs
        """
        return self.loss.grad(d_syn, isrc)

    def _loss_grad_oneshot(self, vp, src, solver, isrc,
                           computeloss=True, computegrad=True) -> Tuple[float, NDArray]:
        """Raw loss function and gradient for one shot

        Compute raw loss function and gradient for one shot without applying any pre/post-processing. Note
        that Devito returns the gradient for slowness square.

        """
        # Compute synthetic data and full forward wavefield u0
        adjsrc, u0, usnaps, _ = solver.forward(vp=vp, save=True if self.factor is None else False,
                                               src=src, autotune=True, factor=self.factor)

        # Compute loss
        if computeloss:
            loss = self.loss(adjsrc.data[:].ravel(), isrc)
        if computegrad:
            # Compute adjoint source
            adjsrc.data[:] = self._adjoint_source(adjsrc.data[:].ravel(), isrc).reshape(adjsrc.data.shape)

            # Compute gradient
            grad, _ = solver.gradient(rec=adjsrc, u=u0, usnaps=usnaps, vp=vp, checkpointing=self.checkpointing,
                                      autotune=True, factor=self.factor)

        if computeloss and computegrad:
            return loss, grad
        elif computeloss:
            return loss
        else:
            return grad

    def _loss_grad(self, vp, isrcs=None, postprocess=None, computeloss=True, computegrad=True):
        """Compute loss function and gradient

        Parameters
        ----------
        vp : :obj:`numpy.ndarray`
            Velocity model in km/s of size ``(nx, ny, nz)``
        isrcs : :obj:`list`, optional
            Indices of shots to be used in gradient computation 
            (if ``None``, use all shots whose number is inferred from ``dobs``)
        postprocess : :obj:`funct`, optional
            Function handle applying postprocessing to gradient and loss
        computeloss : :obj:`bool`, optional
            Compute loss function
        computegrad : :obj:`bool`, optional
            Compute gradient

        Returns
        -------
        loss : :obj:`float`
            Loss function
        grad : :obj:`numpy.ndarray`
            Gradient of size ``(nx, ny, nz)``

        """
        # Create model with class vp to define a geometry and time axis consistent with
        # the observed data and one with provided vp (to be used as input for loss and
        # gradient computation)
        model = self._create_model(self.shape, self.origin, self.spacing,
                                   self.vp, self.space_order, self.nbl, self.fs)
        modelvp = self._create_model(self.shape, self.origin, self.spacing,
                                     vp, self.space_order, self.nbl, self.fs)

        # Identify number of shots
        if isrcs is None:
            nsrc = self.src[0].size
            isrcs = range(nsrc)

        # Initial geometry (source and receivers from the first iterated shot).
        # Both src and rec positions will be mutated inside the loop below.
        isrc0 = next(iter(isrcs)) if not isinstance(isrcs, range) else isrcs[0]
        geometry = self._create_geometry(model,
                                         self.src[0][isrc0:isrc0+1],
                                         self.src[1][isrc0:isrc0+1],
                                         self.src[2][isrc0:isrc0+1],
                                         self.rec[0][isrc0], self.rec[1][isrc0], self.rec[2][isrc0],
                                         self.t0, self.tn, self.src_type, f0=self.f0, dt=self.dt)

        # Re-create source (if wav is not None)
        if self.wav is None:
            src = geometry.src
        else:
            src = CustomSource(name='src', grid=model.grid,
                               wav=self.wav, npoint=1,
                               time_range=geometry.time_axis)

        # Solver
        solver = AcousticWaveSolver(model, geometry,
                                    space_order=self.space_order)

        # Compute loss and gradient
        loss = 0.
        for i, isrc in enumerate(tqdm(isrcs, desc="Computing gradient")):
            # Update source location in geometry
            geometry.src_positions[0, :] = (self.src[0][isrc], self.src[1][isrc], self.src[2][isrc])
            src.coordinates.data[0, :] = (self.src[0][isrc], self.src[1][isrc], self.src[2][isrc])
            # Update receiver locations in geometry (no-op when receivers are
            # shared across shots, since self.rec[k][isrc] is identical for all isrc)
            if self.per_shot_recs:
                geometry.rec_positions[:, 0] = self.rec[0][isrc]
                geometry.rec_positions[:, 1] = self.rec[1][isrc]
                geometry.rec_positions[:, 2] = self.rec[2][isrc]

            # Compute loss and gradient for one shot
            lossgrad = self._loss_grad_oneshot(modelvp.vp, src, solver, isrc,
                                               computeloss=computeloss,
                                               computegrad=computegrad)
            if computeloss and computegrad:
                loss += lossgrad[0]
                if i == 0:
                    grad = lossgrad[1].data[:]
                else:
                    grad += lossgrad[1].data[:]
            elif computeloss:
                loss += lossgrad
            elif computegrad:
                if i == 0:
                    grad = lossgrad.data[:]
                else:
                    grad += lossgrad.data[:]

        if self.clearcache:
            clear_devito_cache()

        # Gather gradients
        if self.base_comm is not None:
            if computeloss:
                loss = self.base_comm.allreduce(loss, op=MPI.SUM)
            if computegrad:
                grad = self.base_comm.allreduce(grad, op=MPI.SUM)

        # Postprocess loss and gradient
        if computegrad:
            grad = self._crop_model(grad, self.nbl, self.fs)
        vp = self._crop_model(modelvp.vp.data[:], self.nbl, self.fs)
        if postprocess is not None:
            if computeloss and computegrad:
                loss, grad = postprocess(vp, loss, grad)
            elif computegrad:
                _, grad = postprocess(vp, None, grad)
            elif computeloss:
                loss, _ = postprocess(vp, loss, None)

        if computeloss and computegrad:
            return loss, grad
        elif computeloss:
            return loss
        else:
            return grad

    def loss_grad(self, x, convertvp=None, postprocess=None,
                  computeloss=True, computegrad=True,
                  debug=False, gradlims=None):
        """Compute loss function and gradient to be used by solver

        This routine wraps _loss_grad providing and returning numpy arrays 
        and should be used with any solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Model obtained by the solver
        convertvp : :obj:`func`, optional
            Function handle that converts the model obtained by the solver in velocity to be used by the propagator
            (if ``None``, it is assumed that the solver itself is working with a velocity model)
        postprocess : :obj:`funct`, optional
            Function handle applying postprocessing to gradient and loss
        computeloss : :obj:`bool`, optional
            Compute loss function
        computegrad : :obj:`bool`, optional
            Compute gradient
        debug : :obj:`bool`, optional
            Debugging flag
        gradlims : :obj:`tuple`, optional
            Limits of gradient to be used in plotting when ``debug=True``

        Returns
        -------
        loss : :obj:`float`
            Loss function
        grad : :obj:`numpy.ndarray`
            Gradient of size ``(nx, ny, nz)``

        """

        # Convert x to velocity
        if convertvp is None:
            vp = x.reshape(self.shape)
        else:
            vp = convertvp(x.reshape(self.shape))

        # Evaluate objective function and gradient
        lossgrad = self._loss_grad(vp.reshape(self.shape),
                                   postprocess=postprocess,
                                   computeloss=computeloss,
                                   computegrad=computegrad)

        # Split lossgrad based on what has been computed in self._loss_grad
        if computeloss and computegrad:
            loss, grad = lossgrad
        elif computeloss:
            loss, grad = lossgrad, None
        else:
            loss, grad = None, lossgrad

        # Save loss history
        if computeloss:
            self.losshistory.append(loss)

        # Display results in debugging mode (central depth slice through grad)
        if debug and computeloss and computegrad:
            print('Debug - loss, grad.min(), grad.max()',
                  loss, grad.min(), grad.max())
            iy = grad.shape[1] // 2
            plt.figure()
            plt.imshow(grad[:, iy, :].T,
                       vmin=gradlims[0] if gradlims is not None else -grad.max(),
                       vmax=gradlims[1] if gradlims is not None else grad.max(),
                       aspect='auto', cmap='seismic')
            plt.colorbar()
            plt.title(f'Gradient (y-slice at iy={iy})')

        # Return loss, grad or both
        if computeloss and computegrad:
            return loss, grad.ravel()
        elif computeloss:
            return loss
        else:
            return grad.ravel()

    def loss(self, x, convertvp=None, postprocess=None):
        """Compute loss function to be used by solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Model obtained by the solver
        convertvp : :obj:`func`, optional
            Function handle that converts the model obtained by the solver in velocity to be used by the propagator
            (if ``None``, it is assumed that the solver itself is working with a velocity model)
        postprocess : :obj:`funct`, optional
            Function handle applying postprocessing to gradient and loss

        Returns
        -------
        loss : :obj:`float`
            Loss function

        """
        return self.loss_grad(x, convertvp=convertvp, postprocess=postprocess,
                              computeloss=True, computegrad=False)

    def grad(self, x, convertvp=None, postprocess=None):
        """Compute gradient to be used by solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Model obtained by the solver
        convertvp : :obj:`func`, optional
            Function handle that converts the model obtained by the solver in velocity to be used by the propagator
            (if ``None``, it is assumed that the solver itself is working with a velocity model)
        postprocess : :obj:`funct`, optional
            Function handle applying postprocessing to gradient and loss

        Returns
        -------
        grad : :obj:`numpy.ndarray`
            Gradient of size ``(nx, ny, nz)``

        """
        return self.loss_grad(x, convertvp=convertvp, postprocess=postprocess,
                              computeloss=False, computegrad=True)
