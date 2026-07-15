document.addEventListener('DOMContentLoaded', () => {
  // --- SELECTORS ---
  const albedoVideoElements = document.querySelectorAll('.albedo-grid-el');
  const albedoSceneButtons = document.querySelectorAll('[data-albedo-scene]');
  const albedoPlayPauseBtn = document.getElementById('albedo-play-pause-btn');

  const videoElements = document.querySelectorAll('.video-grid-el');
  const videoSceneButtons = document.querySelectorAll('[data-video-scene]');
  const relightPlayPauseBtn = document.getElementById('relight-play-pause-btn');

  // --- STATE ---
  let activeAlbedoScene = 'lego';
  let activeVideoScene = 'lego';
  
  let albedoIsPlaying = true;
  let relightIsPlaying = true;

  // --- VIDEO SYNCHRONIZATION FUNCTION ---
  function syncVideoGroup(videos, isPlayingState) {
    let loadedCount = 0;
    const total = videos.length;

    // Reset loaders and pause all
    videos.forEach(video => {
      video.pause();
      const wrapper = video.closest('.video-wrapper');
      if (wrapper) {
        wrapper.classList.remove('loaded');
      }
    });

    const checkReady = () => {
      loadedCount++;
      if (loadedCount >= total) {
        // Play all simultaneously if isPlayingState is true
        videos.forEach(video => {
          const wrapper = video.closest('.video-wrapper');
          if (wrapper) {
            wrapper.classList.add('loaded');
          }
          video.currentTime = 0;
          if (isPlayingState) {
            video.play().catch(e => {
              console.log("Play failed:", e);
            });
          }
        });
      }
    };

    videos.forEach(video => {
      if (video.readyState >= 3) {
        checkReady();
      } else {
        const onReady = () => {
          video.removeEventListener('canplaythrough', onReady);
          video.removeEventListener('loadeddata', onReady);
          checkReady();
        };
        video.addEventListener('canplaythrough', onReady);
        video.addEventListener('loadeddata', onReady);
      }
    });
  }

  // --- UPDATE ALBEDO GRID ---
  function updateAlbedoGrid() {
    const activeGroup = [];
    albedoVideoElements.forEach(video => {
      const model = video.dataset.model;
      const newSrc = `static/videos/${activeAlbedoScene}_${model}_albedo.mp4`;
      
      const source = video.querySelector('source');
      if (source) {
        source.setAttribute('src', newSrc);
      }
      activeGroup.push(video);
      video.load();
    });
    syncVideoGroup(activeGroup, albedoIsPlaying);
  }

  // --- UPDATE RELIGHTING GRID ---
  function updateVideos() {
    const activeGroup = [];
    videoElements.forEach(video => {
      const model = video.dataset.model;
      const env = video.dataset.env;
      const newSrc = `static/videos/${activeVideoScene}_${model}_${env}.mp4`;
      
      const source = video.querySelector('source');
      if (source) {
        source.setAttribute('src', newSrc);
      }
      activeGroup.push(video);
      video.load();
    });
    syncVideoGroup(activeGroup, relightIsPlaying);
  }

  function updatePlayPauseBtnUI(btn, isPlaying) {
    const icon = btn.querySelector('.icon i');
    const text = btn.querySelector('.btn-text');
    if (isPlaying) {
      icon.className = 'fa-solid fa-pause';
      text.textContent = 'Pause';
    } else {
      icon.className = 'fa-solid fa-play';
      text.textContent = 'Play';
    }
  }

  // --- EVENT LISTENERS ---
  
  // Albedo Grid Scene Toggle
  albedoSceneButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      albedoSceneButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeAlbedoScene = btn.dataset.albedoScene;
      updateAlbedoGrid();
    });
  });

  // Albedo Play/Pause Toggle
  if (albedoPlayPauseBtn) {
    albedoPlayPauseBtn.addEventListener('click', () => {
      albedoIsPlaying = !albedoIsPlaying;
      updatePlayPauseBtnUI(albedoPlayPauseBtn, albedoIsPlaying);
      
      albedoVideoElements.forEach(video => {
        if (albedoIsPlaying) {
          video.play().catch(e => console.log("Play failed:", e));
        } else {
          video.pause();
        }
      });
    });
  }

  // Video Grid Scene Toggle
  videoSceneButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      videoSceneButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeVideoScene = btn.dataset.videoScene;
      updateVideos();
    });
  });

  // Relight Play/Pause Toggle
  if (relightPlayPauseBtn) {
    relightPlayPauseBtn.addEventListener('click', () => {
      relightIsPlaying = !relightIsPlaying;
      updatePlayPauseBtnUI(relightPlayPauseBtn, relightIsPlaying);
      
      videoElements.forEach(video => {
        if (relightIsPlaying) {
          video.play().catch(e => console.log("Play failed:", e));
        } else {
          video.pause();
        }
      });
    });
  }

  // --- ACTIVE NAVBAR LINK SCROLL TRACKING ---
  const navItems = document.querySelectorAll('.navbar-end .navbar-item');
  const observedSections = document.querySelectorAll('section[id]');

  const observerOptions = {
    root: null,
    rootMargin: '-20% 0px -60% 0px',
    threshold: 0
  };

  const scrollObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.getAttribute('id');
        
        // Remove active class from all nav items
        navItems.forEach(item => item.classList.remove('active'));
        
        // Map sections to links
        let targetHref = `#${id}`;
        if (id === 'relighting-videos') {
          targetHref = '#albedo-grid';
        }
        
        const activeLink = document.querySelector(`.navbar-end .navbar-item[href="${targetHref}"]`);
        if (activeLink) {
          activeLink.classList.add('active');
        }
      }
    });
  }, observerOptions);

  observedSections.forEach(section => {
    scrollObserver.observe(section);
  });

  // Initial states setup
  setTimeout(() => {
    updateAlbedoGrid();
    updateVideos();
  }, 100);
});
