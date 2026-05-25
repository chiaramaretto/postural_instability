import os
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
import tensorflow as tf
from itertools import combinations


# ──────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION TRAINING (task recognition)
# ──────────────────────────────────────────────────────────────────────────────

def train(model, x_train, y_train, x_val, y_val,
          checkpoint_path, weights_filename="best_cnn_gru.weights.h5"):
    """
    Train CnnGruModel on 3-class task recognition (stance / walking / turning).
    y_train / y_val: integer class labels (0, 1, 2).
    """
    full_path = os.path.join(checkpoint_path, weights_filename)

    # Balanced class weights
    classes = np.unique(y_train)
    cw = class_weight.compute_class_weight("balanced", classes=classes, y=y_train)
    cw_dict = dict(enumerate(cw))
    print(f"Training con pesi classe: {cw_dict}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            full_path, monitor="val_accuracy", mode="max",
            save_best_only=True, save_weights_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=20, restore_best_weights=True,
        ),
    ]

    return model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=64,
        class_weight=cw_dict,
        callbacks=callbacks,
        verbose=1,
    )


# ──────────────────────────────────────────────────────────────────────────────
# AUTOENCODER TRAINING (self-supervised reconstruction)
# ──────────────────────────────────────────────────────────────────────────────

def train_autoencoder(model, x_train, x_val,
                      checkpoint_path, weights_filename="best_ae.weights.h5"):
    """
    Train ImuEncoder as a masked autoencoder.
    Target = original signal (reconstruction objective, MSE loss).
    No class labels needed — fully self-supervised.

    x_train / x_val: float32 arrays of shape (n_windows, 640, 6).
    """
    full_path = os.path.join(checkpoint_path, weights_filename)

    print(f"Autoencoder training: {x_train.shape[0]} train windows, "
          f"{x_val.shape[0]} val windows")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss="mse",
        metrics=["mae"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            full_path, monitor="val_loss", mode="min",
            save_best_only=True, save_weights_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7,
            min_lr=1e-6, verbose=1,
        ),
    ]

    # Input = masked signal (handled inside model.call during training)
    # Target = original signal
    return model.fit(
        x_train, x_train,          # target is the original, masking is inside call()
        validation_data=(x_val, x_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )


def mmd_loss(s_features, t_features, sigma=1.0):
    """Compute RBF-MMD between two feature batches (TensorFlow tensors).

    Returns a scalar tensor representing the MMD estimate.
    """
    def rbf_kernel(x, y):
        x_col = tf.expand_dims(x, 1)  # [n,1,d]
        y_row = tf.expand_dims(y, 0)  # [1,m,d]
        dist_sq = tf.reduce_sum(tf.square(x_col - y_row), axis=-1)
        return tf.exp(-dist_sq / (2.0 * sigma ** 2))

    K_ss = rbf_kernel(s_features, s_features)
    K_tt = rbf_kernel(t_features, t_features)
    K_st = rbf_kernel(s_features, t_features)

    return tf.reduce_mean(K_ss) + tf.reduce_mean(K_tt) - 2.0 * tf.reduce_mean(K_st)


def _median_heuristic_sigma(s_features, t_features, eps=1e-6):
    """Estimate a kernel scale from the median pairwise distance.

    Falls back to 1.0 when the batch is too small or degenerate.
    """
    joined = tf.concat([s_features, t_features], axis=0)

    def compute_median():
        diffs = tf.expand_dims(joined, 1) - tf.expand_dims(joined, 0)
        dist_sq = tf.reduce_sum(tf.square(diffs), axis=-1)
        dist_sq = tf.boolean_mask(dist_sq, dist_sq > 0.0)

        def fallback():
            return tf.constant(1.0, dtype=joined.dtype)

        def from_distances():
            sorted_dist = tf.sort(dist_sq)
            mid = tf.shape(sorted_dist)[0] // 2
            median_sq = tf.cond(
                tf.equal(tf.shape(sorted_dist)[0] % 2, 0),
                lambda: (sorted_dist[mid - 1] + sorted_dist[mid]) / 2.0,
                lambda: sorted_dist[mid],
            )
            median = tf.sqrt(tf.maximum(median_sq, eps))
            return tf.maximum(median, tf.constant(1.0, dtype=joined.dtype) * eps)

        return tf.cond(tf.equal(tf.size(dist_sq), 0), fallback, from_distances)

    return tf.cond(tf.less(tf.shape(joined)[0], 2), lambda: tf.constant(1.0, dtype=joined.dtype), compute_median)


def mmd_loss_multiscale(s_features, t_features, sigmas=None):
    """Compute MMD using a multiscale RBF kernel.

    If sigmas is None, the kernel scales are built around the median heuristic.
    """
    if sigmas is None:
        median_sigma = _median_heuristic_sigma(s_features, t_features)
        sigmas = [median_sigma * 0.5, median_sigma, median_sigma * 2.0]

    total = 0.0
    for sigma in sigmas:
        total += mmd_loss(s_features, t_features, sigma=sigma)
    return total


def normalize_latents(latents):
    return tf.nn.l2_normalize(latents, axis=-1)


def _as_group_list(data):
    if data is None:
        return []
    if isinstance(data, dict):
        groups = list(data.values())
    elif isinstance(data, (list, tuple)):
        groups = list(data)
    else:
        groups = [data]
    return [np.asarray(group) for group in groups if len(group) > 0]


def _batch_iter(array, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(array).shuffle(1024).batch(batch_size).repeat()
    return iter(dataset)


def _pairwise_multiscale_mmd(latent_batches, sigma):
    if len(latent_batches) < 2:
        return tf.constant(0.0, dtype=latent_batches[0].dtype)

    pairwise_terms = []
    for first_idx, second_idx in combinations(range(len(latent_batches)), 2):
        first = latent_batches[first_idx]
        second = latent_batches[second_idx]
        pairwise_terms.append(
            mmd_loss_multiscale(
                first,
                second,
                sigmas=None if sigma is None else [sigma / 10.0, sigma, sigma * 10.0],
            )
        )

    return tf.add_n(pairwise_terms) / tf.cast(len(pairwise_terms), tf.float32)


def _batched_dataset(array, batch_size):
    return tf.data.Dataset.from_tensor_slices(array).batch(batch_size)


def _collect_normalized_latents(model, arrays, batch_size, training=False):
    collected = []
    for array in arrays:
        batches = []
        for batch in _batched_dataset(array, batch_size):
            try:
                latent = model.get_latent(batch, training=training)
            except AttributeError:
                latent = model.encoder(batch, training=training)
            batches.append(normalize_latents(latent))
        collected.append(tf.concat(batches, axis=0))
    return collected


def _mean_reconstruction_loss(model, arrays, batch_size):
    losses = []
    for array in arrays:
        batch_losses = []
        for batch in _batched_dataset(array, batch_size):
            recon = model(batch, training=False)
            batch_losses.append(tf.reduce_mean(tf.square(recon - batch)))
        losses.append(tf.add_n(batch_losses) / tf.cast(len(batch_losses), tf.float32))
    return tf.add_n(losses) / tf.cast(len(losses), tf.float32)


def _mean_classification_loss(model, arrays_x, arrays_y, loss_fn, batch_size):
    losses = []
    for x_array, y_array in zip(arrays_x, arrays_y):
        batch_losses = []
        for x_batch, y_batch in _batched_dataset((x_array, y_array), batch_size):
            preds = model(x_batch, training=False)
            batch_losses.append(loss_fn(y_batch, preds))
        losses.append(tf.add_n(batch_losses) / tf.cast(len(batch_losses), tf.float32))
    return tf.add_n(losses) / tf.cast(len(losses), tf.float32)


def train_autoencoder_mmd(model, x_source, x_target, x_val_source, x_val_target,
                          checkpoint_path, weights_filename="best_ae_mmd.weights.h5",
                          lambda_mmd=0.01, sigma=None, epochs=100, batch_size=64,
                          learning_rate=3e-4, clipnorm=1.0):
    """Train an autoencoder with an added MMD penalty between source and target encodings.

    x_source / x_target are numpy arrays of windows from source and target domains.
    Validation arrays are optional but recommended (pass empty arrays to skip val check).
    """
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)

    source_groups = _as_group_list(x_source)
    target_groups = _as_group_list(x_target)

    multi_domain = len(source_groups) > 1 and len(target_groups) == 0
    if multi_domain:
        domain_groups = source_groups
        domain_iters = [_batch_iter(group, batch_size) for group in domain_groups]
        steps_per_epoch = max(1, min(len(group) for group in domain_groups) // batch_size)
    else:
        source_array = np.asarray(source_groups[0])
        target_array = np.asarray(target_groups[0]) if target_groups else source_array[1::2]
        ds_s = tf.data.Dataset.from_tensor_slices(source_array).shuffle(1024).batch(batch_size).repeat()
        ds_t = tf.data.Dataset.from_tensor_slices(target_array).shuffle(1024).batch(batch_size).repeat()
        paired = tf.data.Dataset.zip((ds_s, ds_t))
        steps_per_epoch = max(1, min(len(source_array), len(target_array)) // batch_size)

    # Simple checkpointing: save best val loss (reconstruction + lambda * mmd)
    best_val = float('inf')
    os.makedirs(checkpoint_path, exist_ok=True)
    full_path = os.path.join(checkpoint_path, weights_filename)

    def train_step(x_s, x_t):
        with tf.GradientTape() as tape:
            # reconstruction
            recon_s = model(x_s, training=True)
            recon_t = model(x_t, training=True)
            loss_rec = tf.reduce_mean(tf.square(recon_s - x_s)) + tf.reduce_mean(tf.square(recon_t - x_t))

            # latent extraction: assume model has method `get_latent`
            try:
                z_s = normalize_latents(model.get_latent(x_s, training=True))
                z_t = normalize_latents(model.get_latent(x_t, training=True))
            except AttributeError:
                # fallback: if encoder is accessible as model.encoder
                z_s = normalize_latents(model.encoder(x_s, training=True))
                z_t = normalize_latents(model.encoder(x_t, training=True))

            loss_m = mmd_loss_multiscale(z_s, z_t, sigmas=None if sigma is None else [sigma / 10.0, sigma, sigma * 10.0])
            total = loss_rec + lambda_mmd * loss_m

        grads = tape.gradient(total, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return float(loss_rec.numpy()), float(loss_m.numpy()), float(total.numpy())

    # Validation helper
    def val_loss():
        val_source_groups = _as_group_list(x_val_source)
        val_target_groups = _as_group_list(x_val_target)

        if len(val_source_groups) == 0:
            return float('inf')
        if multi_domain:
            latent_terms = _collect_normalized_latents(model, val_source_groups, batch_size=batch_size, training=False)
            loss_rec_v = float(_mean_reconstruction_loss(model, val_source_groups, batch_size=batch_size).numpy())
            loss_m_v = float(_pairwise_multiscale_mmd(latent_terms, sigma).numpy())
            return loss_rec_v + lambda_mmd * loss_m_v

        if len(val_target_groups) == 0:
            return float('inf')

        x_val_source_arr = val_source_groups[0]
        x_val_target_arr = val_target_groups[0]
        zs = _collect_normalized_latents(model, [x_val_source_arr], batch_size=batch_size, training=False)[0]
        zt = _collect_normalized_latents(model, [x_val_target_arr], batch_size=batch_size, training=False)[0]
        rec_s = float(_mean_reconstruction_loss(model, [x_val_source_arr], batch_size=batch_size).numpy())
        rec_t = float(_mean_reconstruction_loss(model, [x_val_target_arr], batch_size=batch_size).numpy())
        loss_rec_v = rec_s + rec_t
        loss_m_v = float(mmd_loss_multiscale(zs, zt, sigmas=None if sigma is None else [sigma / 10.0, sigma, sigma * 10.0]).numpy())
        return loss_rec_v + lambda_mmd * loss_m_v

    if multi_domain:
        iters = domain_iters
    else:
        it = iter(paired)
    for ep in range(1, epochs + 1):
        epoch_rec = 0.0
        epoch_mmd = 0.0
        epoch_tot = 0.0
        for step in range(steps_per_epoch):
            if multi_domain:
                batches = [next(domain_iter) for domain_iter in iters]
                with tf.GradientTape() as tape:
                    rec_terms = []
                    latent_batches = []
                    for batch in batches:
                        recon = model(batch, training=True)
                        rec_terms.append(tf.reduce_mean(tf.square(recon - batch)))
                        try:
                            latent_batches.append(normalize_latents(model.get_latent(batch, training=True)))
                        except AttributeError:
                            latent_batches.append(normalize_latents(model.encoder(batch, training=True)))

                    loss_rec = tf.add_n(rec_terms) / tf.cast(len(rec_terms), tf.float32)
                    loss_m = _pairwise_multiscale_mmd(latent_batches, sigma)
                    total = loss_rec + lambda_mmd * loss_m

                grads = tape.gradient(total, model.trainable_variables)
                opt.apply_gradients(zip(grads, model.trainable_variables))
                r = float(loss_rec.numpy())
                m = float(loss_m.numpy())
                tot = float(total.numpy())
            else:
                x_s_batch, x_t_batch = next(it)
                r, m, tot = train_step(x_s_batch, x_t_batch)
            epoch_rec += r
            epoch_mmd += m
            epoch_tot += tot

        epoch_rec /= steps_per_epoch
        epoch_mmd /= steps_per_epoch
        epoch_tot /= steps_per_epoch

        # compute val and checkpoint
        v = val_loss()
        print(f"Epoch {ep}/{epochs} — rec: {epoch_rec:.6f}, mmd: {epoch_mmd:.6f}, total: {epoch_tot:.6f}, val_total: {v:.6f}")
        if v < best_val:
            best_val = v
            model.save_weights(full_path)
            print(f"Saved best weights -> {full_path}")

    # restore best if available
    if os.path.exists(full_path):
        model.load_weights(full_path)

    return model


def train_classifier_mmd(model, x_source, y_source, x_target,
                          x_val_source=None, y_val_source=None,
                          checkpoint_path=None, weights_filename="best_clf_mmd.weights.h5",
                          lambda_mmd=0.1, sigma=None, epochs=50, batch_size=64,
                          learning_rate=1e-4, clipnorm=1.0):
    """Train a classifier model minimizing classification loss on source
    plus an MMD penalty between source and target latents.

    x_target may be unlabeled; only used to compute MMD.
    """
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    source_groups = _as_group_list(x_source)
    label_groups = _as_group_list(y_source)
    target_groups = _as_group_list(x_target)

    multi_domain = len(source_groups) > 1 and len(label_groups) > 1 and len(target_groups) == 0
    if multi_domain:
        if len(source_groups) != len(label_groups):
            raise ValueError("When passing multiple source domains, x_source and y_source must have the same number of groups.")
        domain_groups = source_groups
        domain_label_groups = label_groups
        domain_iters = [
            iter(tf.data.Dataset.from_tensor_slices((x_group, y_group)).shuffle(1024).batch(batch_size).repeat())
            for x_group, y_group in zip(domain_groups, domain_label_groups)
        ]
        steps_per_epoch = max(1, min(len(group) for group in domain_groups) // batch_size)
    else:
        source_array = np.asarray(source_groups[0])
        label_array = np.asarray(label_groups[0]) if label_groups else np.asarray(y_source)
        if len(target_groups) == 0:
            target_groups = [source_array[1::2] if len(source_array) > 1 else source_array]

        ds_s = tf.data.Dataset.from_tensor_slices((source_array, label_array)).shuffle(1024).batch(batch_size).repeat()
        ds_targets = [tf.data.Dataset.from_tensor_slices(group).shuffle(1024).batch(batch_size).repeat() for group in target_groups]
        iters_targets = [iter(ds_t) for ds_t in ds_targets]

        min_target_len = min(len(group) for group in target_groups)
        steps_per_epoch = max(1, min(len(source_array), min_target_len) // batch_size)

    os.makedirs(checkpoint_path or '.', exist_ok=True)
    full_path = os.path.join(checkpoint_path or '.', weights_filename)
    best_val = float('inf')

    def train_step(x_s, y_s, x_t):
        with tf.GradientTape() as tape:
            logits = model(x_s, training=True)
            cls_loss = loss_fn(y_s, logits)

            z_s = normalize_latents(model.get_latent(x_s, training=True))
            z_t = normalize_latents(model.get_latent(x_t, training=True))
            loss_m = mmd_loss_multiscale(z_s, z_t, sigmas=None if sigma is None else [sigma / 10.0, sigma, sigma * 10.0])

            total = cls_loss + lambda_mmd * loss_m

        grads = tape.gradient(total, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return float(cls_loss.numpy()), float(loss_m.numpy()), float(total.numpy())

    if multi_domain:
        iters = domain_iters
    else:
        it = iter(ds_s)
    for ep in range(1, epochs + 1):
        epoch_cls = 0.0
        epoch_mmd = 0.0
        epoch_tot = 0.0
        for step in range(steps_per_epoch):
            if multi_domain:
                batches = [next(domain_iter) for domain_iter in iters]
                cls_losses = []
                latent_batches = []
                with tf.GradientTape() as tape:
                    for x_batch, y_batch in batches:
                        logits = model(x_batch, training=True)
                        cls_losses.append(loss_fn(y_batch, logits))
                        latent_batches.append(normalize_latents(model.get_latent(x_batch, training=True)))

                    cls_loss = tf.add_n(cls_losses) / tf.cast(len(cls_losses), tf.float32)
                    loss_m = _pairwise_multiscale_mmd(latent_batches, sigma)
                    total = cls_loss + lambda_mmd * loss_m

                grads = tape.gradient(total, model.trainable_variables)
                opt.apply_gradients(zip(grads, model.trainable_variables))

                c = float(cls_loss.numpy())
                m = float(loss_m.numpy())
                tot = float(total.numpy())
            else:
                x_s_batch, y_s_batch = next(it)
                x_t_batches = [next(target_iter) for target_iter in iters_targets]

                with tf.GradientTape() as tape:
                    logits = model(x_s_batch, training=True)
                    cls_loss = loss_fn(y_s_batch, logits)

                    z_s = normalize_latents(model.get_latent(x_s_batch, training=True))
                    mmd_terms = []
                    for x_t_batch in x_t_batches:
                        z_t = normalize_latents(model.get_latent(x_t_batch, training=True))
                        mmd_terms.append(mmd_loss_multiscale(z_s, z_t, sigmas=None if sigma is None else [sigma / 10.0, sigma, sigma * 10.0]))

                    loss_m = tf.add_n(mmd_terms) / tf.cast(len(mmd_terms), tf.float32)
                    total = cls_loss + lambda_mmd * loss_m

                grads = tape.gradient(total, model.trainable_variables)
                opt.apply_gradients(zip(grads, model.trainable_variables))

                c = float(cls_loss.numpy())
                m = float(loss_m.numpy())
                tot = float(total.numpy())
            epoch_cls += c
            epoch_mmd += m
            epoch_tot += tot

        epoch_cls /= steps_per_epoch
        epoch_mmd /= steps_per_epoch
        epoch_tot /= steps_per_epoch

        # simple val check on source validation if provided
        if x_val_source is not None and y_val_source is not None and len(x_val_source) > 0:
            if multi_domain:
                val_loss = float(_mean_classification_loss(model, _as_group_list(x_val_source), _as_group_list(y_val_source), loss_fn, batch_size=batch_size).numpy())
            else:
                val_x = np.asarray(x_val_source)
                val_y = np.asarray(y_val_source)
                val_batches = _batched_dataset((val_x, val_y), batch_size)
                batch_losses = []
                for x_batch, y_batch in val_batches:
                    preds = model(x_batch, training=False)
                    batch_losses.append(loss_fn(y_batch, preds))
                val_loss = float((tf.add_n(batch_losses) / tf.cast(len(batch_losses), tf.float32)).numpy())
        else:
            val_loss = float('inf')

        print(f"Epoch {ep}/{epochs} — cls: {epoch_cls:.6f}, mmd: {epoch_mmd:.6f}, total: {epoch_tot:.6f}, val_cls: {val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            model.save_weights(full_path)
            print(f"Saved best classifier weights -> {full_path}")

    if os.path.exists(full_path):
        model.load_weights(full_path)

    return model