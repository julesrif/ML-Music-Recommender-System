<script setup lang="ts">
import { ref, computed, watch, onMounted, nextTick } from 'vue'

type Track = {
  track_id: string
  name: string
  artists: string
  genres: string
  tier?: string
  placement?: string
  score?: number
}

// A cold track has no learned vector of its own, so the badge states what stood in for it.
const TIER_STYLE: Record<string, { label: string; class: string }> = {
  learned: { label: 'trained', class: 'bg-emerald-950 text-emerald-400 border-emerald-800' },
  artist: { label: 'unseen · via artist', class: 'bg-amber-950 text-amber-400 border-amber-800' },
  genre: { label: 'unseen · via genre', class: 'bg-rose-950 text-rose-400 border-rose-900' },
  catalogue: { label: 'unseen · no signal', class: 'bg-slate-800 text-slate-400 border-slate-700' },
}

function tierOf(track: Track) {
  return TIER_STYLE[track.tier ?? 'learned'] ?? TIER_STYLE.catalogue
}

type Artist = { name: string; track_count: number }
type Genre = { name: string; artist_count: number }

const ARTIST_PAGE = 50

const genres = ref<Genre[]>([])
const genreQuery = ref('')
const filteredGenres = computed(() => {
  if (!genreQuery.value) return genres.value
  const q = genreQuery.value.toLowerCase()
  return genres.value.filter(g => g.name.toLowerCase().includes(q))
})
const selectedGenres = ref<string[]>([])
const artistSort = ref<'plays' | 'name'>('plays')

const artistQuery = ref('')
const artists = ref<Artist[]>([])
const artistTotal = ref(0)
const artistOffset = ref(0)
const artistHasMore = ref(false)
const loadingArtists = ref(false)
const loadingMoreArtists = ref(false)

const selectedArtist = ref<Artist | null>(null)
const artistTracks = ref<Track[]>([])
const loadingTracks = ref(false)

const trackQuery = ref('')
const trackResults = ref<Track[]>([])
const trackTotal = ref(0)
const searchingTracks = ref(false)

// A track search overrides the artist selection rather than clearing it, so dismissing the
// search returns the user to whatever artist they were browsing.
const searchMode = computed(() => trackQuery.value.trim().length > 0)
const visibleTracks = computed(() => (searchMode.value ? trackResults.value : artistTracks.value))

const playlist = ref<Track[]>([])
const recommendations = ref<Track[]>([])
const loading = ref(false)
const error = ref('')
const stats = ref<{
  tracks: number
  artists: number
  trained_tracks: number
  tiers: Record<string, number>
} | null>(null)

const spotifyData = ref<Record<string, { cover?: string, preview_url?: string, artist_cover?: string }>>({})

const currentPreviewUrl = ref<string | null>(null)
const audioPlayer = new Audio()
audioPlayer.addEventListener('ended', () => { currentPreviewUrl.value = null })

function togglePreview(url: string | undefined, e: Event) {
  e.stopPropagation()
  if (!url) return
  if (currentPreviewUrl.value === url) {
    audioPlayer.pause()
    currentPreviewUrl.value = null
  } else {
    audioPlayer.src = url
    audioPlayer.play()
    currentPreviewUrl.value = url
  }
}

async function fetchCovers(trackIds: string[]) {
  const missingIds = trackIds.filter(id => !spotifyData.value[id])
  if (missingIds.length === 0) return
  
  try {
    for (let i = 0; i < missingIds.length; i += 50) {
      const chunk = missingIds.slice(i, i + 50)
      const res = await fetch(`/api/covers?ids=${chunk.join(',')}`)
      if (!res.ok) continue
      const data = await res.json()
      if (data.error) console.error("Spotify API Error from Backend:", data.error)
      Object.assign(spotifyData.value, data.data)
    }
  } catch (e) {
    console.error("Failed to fetch covers:", e)
  }
}

const inPlaylist = computed(() => new Set(playlist.value.map((t) => t.track_id)))

async function loadArtists(q: string, append = false) {
  const offset = append ? artistOffset.value : 0
  if (append) loadingMoreArtists.value = true
  else loadingArtists.value = true

  try {
    const params = new URLSearchParams({
      q,
      limit: String(ARTIST_PAGE),
      offset: String(offset),
      genres: selectedGenres.value.join(','),
      sort: artistSort.value,
    })
    const res = await fetch(`/api/artists?${params}`)
    if (!res.ok) throw new Error('Could not load artists')
    const data = await res.json()
    artists.value = append ? [...artists.value, ...data.results] : data.results
    artistTotal.value = data.total
    artistHasMore.value = data.has_more
    artistOffset.value = offset + data.results.length
  } catch (e) {
    error.value = (e as Error).message
  } finally {
    loadingArtists.value = false
    loadingMoreArtists.value = false
  }
}

function toggleGenre(name: string) {
  selectedGenres.value = selectedGenres.value.includes(name)
    ? selectedGenres.value.filter((g) => g !== name)
    : [...selectedGenres.value, name]
}

watch(selectedGenres, () => loadArtists(artistQuery.value))
watch(artistSort, () => loadArtists(artistQuery.value))

function onArtistScroll(event: Event) {
  if (!artistHasMore.value || loadingArtists.value || loadingMoreArtists.value) return
  const el = event.target as HTMLElement
  if (el.scrollTop + el.clientHeight >= el.scrollHeight - 120) {
    loadArtists(artistQuery.value, true)
  }
}

onMounted(async () => {
  loadArtists('')
  try {
    genres.value = (await (await fetch('/api/genres')).json()).results
  } catch {
    /* the genre filter is optional; the artist list still works without it */
  }
  try {
    stats.value = await (await fetch('/api/health')).json()
  } catch {
    /* header stats are cosmetic */
  }
})

let timer: number | undefined
watch(artistQuery, (value) => {
  window.clearTimeout(timer)
  timer = window.setTimeout(() => loadArtists(value), 250)
})

async function searchTracks(q: string) {
  if (!q.trim()) {
    trackResults.value = []
    trackTotal.value = 0
    return
  }
  searchingTracks.value = true
  try {
    const res = await fetch(`/api/tracks?q=${encodeURIComponent(q)}&limit=50`)
    if (!res.ok) throw new Error('Could not search tracks')
    const data = await res.json()
    trackResults.value = data.results
    trackTotal.value = data.total
    fetchCovers(trackResults.value.map(t => t.track_id))
  } catch (e) {
    error.value = (e as Error).message
  } finally {
    searchingTracks.value = false
  }
}

let trackTimer: number | undefined
watch(trackQuery, (value) => {
  window.clearTimeout(trackTimer)
  trackTimer = window.setTimeout(() => searchTracks(value), 250)
})

async function openArtist(artist: Artist) {
  selectedArtist.value = artist
  artistTracks.value = []
  loadingTracks.value = true
  try {
    const res = await fetch(`/api/artists/tracks?name=${encodeURIComponent(artist.name)}`)
    if (!res.ok) throw new Error('Could not load tracks')
    artistTracks.value = (await res.json()).results
    fetchCovers(artistTracks.value.map(t => t.track_id))
  } catch (e) {
    error.value = (e as Error).message
  } finally {
    loadingTracks.value = false
  }
}

function toggleTrack(track: Track) {
  if (inPlaylist.value.has(track.track_id)) {
    playlist.value = playlist.value.filter((t) => t.track_id !== track.track_id)
  } else {
    playlist.value.push(track)
  }
}

async function recommend(excludeArtists = false) {
  if (!playlist.value.length) return
  loading.value = true
  error.value = ''
  recommendations.value = []
  try {
    const res = await fetch('/api/recommend/new', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        track_ids: playlist.value.map((t) => t.track_id), 
        k: 10,
        exclude_artists: excludeArtists
      }),
    })
    if (!res.ok) throw new Error((await res.json()).detail ?? 'Request failed')
    recommendations.value = (await res.json()).recommendations
    fetchCovers(recommendations.value.map(t => t.track_id))
    
    // Auto-scroll to recommendations
    nextTick(() => {
      const el = document.getElementById('recommendations-section')
      if (el) el.scrollIntoView({ behavior: 'smooth' })
    })
  } catch (e) {
    error.value = (e as Error).message
  } finally {
    loading.value = false
  }
}

function reset() {
  playlist.value = []
  recommendations.value = []
  error.value = ''
}
</script>

<template>
  <div class="min-h-screen bg-slate-950 text-slate-100 p-4 sm:p-8 font-sans selection:bg-emerald-500/30">
    <div class="max-w-7xl mx-auto">
      <!-- Hero Section -->
      <header class="mb-10 text-center sm:text-left">
        <h1 class="text-4xl sm:text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400 mb-4 tracking-tight">
          Spotify Playlist Continuation
        </h1>
        <p class="text-slate-400 text-lg max-w-2xl font-light">
          Craft your seed playlist by picking artists and tracks. Our two-tower deep learning model will analyze your selection and seamlessly predict what comes next.
        </p>
        <div v-if="stats" class="mt-4 inline-flex items-center gap-3 px-4 py-2 rounded-full bg-slate-900/50 border border-slate-800 text-xs text-slate-400 backdrop-blur-sm shadow-sm">
          <span class="flex items-center gap-1.5">
            <div class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
            {{ stats.tracks.toLocaleString() }} Tracks
          </span>
          <span class="w-px h-3 bg-slate-700"></span>
          <span>{{ stats.artists.toLocaleString() }} Artists</span>
        </div>
      </header>

      <div class="grid gap-8 lg:grid-cols-12 items-start">
        <!-- Main Content Area (Artists & Tracks) -->
        <div class="lg:col-span-8 space-y-6">
          
          <!-- Artist Selection -->
          <section class="glass-panel rounded-3xl p-6 shadow-2xl relative overflow-hidden group transition-all duration-300 hover:border-slate-600/50">
            <div class="absolute inset-0 bg-gradient-to-br from-emerald-500/5 to-transparent pointer-events-none"></div>
            
            <header class="flex flex-col sm:flex-row sm:items-center justify-between mb-6 relative z-10 gap-4">
              <div>
                <h2 class="text-xl font-semibold flex items-center gap-2">
                  <span class="bg-emerald-500 text-slate-950 w-6 h-6 rounded-full flex items-center justify-center text-sm font-bold shadow-lg shadow-emerald-500/20">1</span>
                  Select an Artist
                </h2>
                <p class="text-sm text-slate-400 mt-1">Search for an artist to reveal their top tracks.</p>
              </div>
              
              <!-- Search & Filter Controls -->
              <div class="flex items-center gap-3 w-full sm:w-auto">
                <div class="relative w-full sm:w-64">
                  <svg class="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                  <input
                    v-model="artistQuery"
                    type="text"
                    placeholder="Search artists..."
                    class="w-full pl-9 pr-4 py-2.5 rounded-xl bg-slate-950/50 border border-slate-700/80 focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 focus:outline-none text-sm transition-all shadow-inner placeholder-slate-600"
                  />
                </div>
              </div>
            </header>

            <!-- Genres -->
            <div class="mb-5 relative z-10">
              <div class="flex items-center justify-between mb-2">
                <span class="text-xs font-medium text-slate-500 uppercase tracking-wider">Genres</span>
                <div class="flex items-center gap-3">
                  <input
                    v-model="genreQuery"
                    type="text"
                    placeholder="Filter genres..."
                    class="px-2 py-1 text-xs rounded bg-slate-900 border border-slate-700 text-slate-300 focus:outline-none focus:border-emerald-500 w-32 sm:w-auto"
                  />
                  <button
                    v-if="selectedGenres.length"
                    class="text-xs text-emerald-400 hover:text-emerald-300 transition-colors"
                    @click="selectedGenres = []"
                  >
                    Clear filters
                  </button>
                </div>
              </div>
              <div class="flex flex-wrap gap-2 max-h-24 overflow-y-auto custom-scrollbar pr-2">
                <button
                  v-for="genre in filteredGenres"
                  :key="genre.name"
                  class="px-3 py-1.5 rounded-lg border text-[11px] font-medium transition-all duration-200 shadow-sm"
                  :class="
                    selectedGenres.includes(genre.name)
                      ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-300'
                      : 'bg-slate-800/40 border-slate-700/50 text-slate-400 hover:bg-slate-700/80 hover:text-slate-200 hover:border-slate-600'
                  "
                  :title="`${genre.artist_count.toLocaleString()} artists`"
                  @click="toggleGenre(genre.name)"
                >
                  {{ genre.name }}
                </button>
              </div>
            </div>

            <!-- Artist List -->
            <div class="relative z-10">
              <div class="flex items-center justify-between text-xs text-slate-500 mb-3 px-1">
                <span>{{ loadingArtists ? 'Searching...' : `${artistTotal.toLocaleString()} artists found` }}</span>
                <div class="flex items-center bg-slate-900 rounded-lg p-0.5 border border-slate-700/50 shadow-inner">
                  <button
                    v-for="option in [{ key: 'plays', label: 'Popular' }, { key: 'name', label: 'A-Z' }]"
                    :key="option.key"
                    class="px-3 py-1 rounded-md transition-all duration-200"
                    :class="artistSort === option.key ? 'bg-slate-700 text-slate-100 shadow' : 'text-slate-400 hover:text-slate-200'"
                    @click="artistSort = option.key as 'plays' | 'name'"
                  >
                    {{ option.label }}
                  </button>
                </div>
              </div>
              
              <ul
                class="max-h-64 overflow-y-auto custom-scrollbar space-y-1 pr-2"
                @scroll="onArtistScroll"
              >
                <li v-for="artist in artists" :key="artist.name">
                  <button
                    class="w-full text-left py-2.5 px-3 rounded-xl transition-all duration-200 flex items-center gap-3 border border-transparent"
                    :class="selectedArtist?.name === artist.name ? 'bg-emerald-500/10 border-emerald-500/30 shadow-sm' : 'hover:bg-slate-800/40 hover:border-slate-700/50'"
                    @click="openArtist(artist)"
                  >
                    <div class="w-10 h-10 rounded-full bg-slate-800 flex items-center justify-center shrink-0 border border-slate-700 shadow-inner">
                      <svg class="w-5 h-5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                      </svg>
                    </div>
                    <div class="flex-1 min-w-0">
                      <p class="text-sm font-semibold truncate transition-colors" :class="selectedArtist?.name === artist.name ? 'text-emerald-400' : 'text-slate-200'">{{ artist.name }}</p>
                      <p class="text-xs text-slate-500 mt-0.5">{{ artist.track_count }} tracks</p>
                    </div>
                  </button>
                </li>
                <li v-if="loadingMoreArtists" class="py-4 flex justify-center">
                  <svg class="w-6 h-6 text-emerald-500 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                </li>
                <li v-if="!loadingArtists && !artists.length" class="py-8 text-center text-sm text-slate-400">
                  <div class="bg-slate-900/50 rounded-xl p-6 inline-block border border-slate-800 shadow-inner">
                    <svg class="w-8 h-8 mx-auto text-slate-600 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                    No artists found matching your criteria.
                  </div>
                </li>
              </ul>
            </div>
          </section>

          <!-- Track Selection -->
          <section class="glass-panel rounded-3xl p-6 shadow-2xl relative overflow-hidden transition-all duration-300 hover:border-slate-600/50">
            <div class="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-transparent pointer-events-none"></div>
            
            <header class="flex flex-col sm:flex-row sm:items-center justify-between mb-6 relative z-10 gap-4">
              <div>
                <h2 class="text-xl font-semibold flex items-center gap-2">
                  <span class="bg-cyan-500 text-slate-950 w-6 h-6 rounded-full flex items-center justify-center text-sm font-bold shadow-lg shadow-cyan-500/20">2</span>
                  Add Tracks
                </h2>
                <p class="text-sm text-slate-400 mt-1">
                  <template v-if="selectedArtist && !searchMode">Top tracks by <strong class="text-slate-300">{{ selectedArtist.name }}</strong></template>
                  <template v-else>Search directly for any track across the catalogue.</template>
                </p>
              </div>
              
              <div class="relative w-full sm:w-64">
                <svg class="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                </svg>
                <input
                  v-model="trackQuery"
                  type="text"
                  placeholder="Find a specific song..."
                  class="w-full pl-9 pr-4 py-2.5 rounded-xl bg-slate-950/50 border border-slate-700/80 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 focus:outline-none text-sm transition-all shadow-inner placeholder-slate-600"
                />
              </div>
            </header>

            <div class="relative z-10">
              <div v-if="searchingTracks || loadingTracks" class="flex justify-center py-12">
                <svg class="w-8 h-8 text-cyan-500 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              </div>
              
              <ul v-else-if="visibleTracks.length > 0" class="max-h-72 overflow-y-auto custom-scrollbar space-y-2 pr-2">
                <li v-for="track in visibleTracks" :key="track.track_id" class="group/track">
                  <div 
                    class="p-3.5 rounded-2xl border transition-all duration-300 flex items-center gap-4 cursor-pointer"
                    :class="inPlaylist.has(track.track_id) ? 'bg-slate-800/90 border-slate-700 shadow-sm' : 'bg-slate-900/40 border-slate-800/80 hover:bg-slate-800 hover:border-slate-600 hover:shadow-md'"
                    @click="toggleTrack(track)"
                  >
                    <button 
                      class="shrink-0 w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-300 shadow-sm overflow-hidden relative group/cover"
                      :class="inPlaylist.has(track.track_id) ? 'bg-slate-700 text-slate-300 hover:bg-rose-500/20 hover:text-rose-400' : 'bg-slate-950 border border-slate-700/80 text-slate-400 group-hover/track:bg-cyan-500 group-hover/track:border-cyan-400 group-hover/track:text-slate-950 group-hover/track:shadow-cyan-500/30 group-hover/track:shadow-lg'"
                    >
                      <img v-if="spotifyData[track.track_id]?.cover" :src="spotifyData[track.track_id].cover" class="absolute inset-0 w-full h-full object-cover transition-opacity duration-300" :class="spotifyData[track.track_id]?.preview_url ? 'group-hover/cover:opacity-40' : ''" />
                      <svg v-else-if="inPlaylist.has(track.track_id)" class="w-6 h-6 relative z-10" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 12H4" /></svg>
                      <svg v-else class="w-6 h-6 relative z-10" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" /></svg>
                      
                      <div v-if="spotifyData[track.track_id]?.preview_url" 
                           class="absolute inset-0 flex items-center justify-center opacity-0 group-hover/cover:opacity-100 transition-opacity bg-black/40 text-white z-20 cursor-pointer"
                           @click.stop="togglePreview(spotifyData[track.track_id]?.preview_url, $event)"
                      >
                        <svg v-if="currentPreviewUrl === spotifyData[track.track_id].preview_url" class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
                        <svg v-else class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                      </div>
                    </button>
                    
                    <div class="flex-1 min-w-0">
                      <p class="text-[15px] font-semibold truncate transition-colors" :class="inPlaylist.has(track.track_id) ? 'text-slate-300' : 'text-slate-100 group-hover/track:text-white'">{{ track.name }}</p>
                      <div class="text-xs text-slate-400 truncate mt-1 flex items-center gap-1.5">
                        <template v-if="searchMode">
                          <img v-if="spotifyData[track.track_id]?.artist_cover" :src="spotifyData[track.track_id].artist_cover" class="w-4 h-4 rounded-full object-cover" />
                          <span>{{ track.artists }}</span>
                        </template>
                        <template v-else>
                          <span>{{ track.genres }}</span>
                        </template>
                      </div>
                    </div>
                    
                    <div class="shrink-0 hidden sm:block">
                      <span
                        class="px-2.5 py-1 rounded-lg border text-[10px] font-bold uppercase tracking-widest shadow-sm"
                        :class="tierOf(track).class"
                        :title="track.placement"
                      >
                        {{ tierOf(track).label }}
                      </span>
                    </div>
                  </div>
                </li>
              </ul>
              
              <div v-else class="py-16 text-center text-slate-500 bg-slate-900/30 rounded-2xl border border-slate-800/50 border-dashed shadow-inner">
                <svg class="w-12 h-12 mx-auto text-slate-700 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                </svg>
                <p class="text-sm font-medium" v-if="searchMode">No tracks found for "{{ trackQuery }}".</p>
                <p class="text-sm font-medium" v-else>Select an artist above to see their tracks.</p>
              </div>
            </div>
          </section>
        </div>

        <!-- Sidebar (Playlist & Results) -->
        <div class="lg:col-span-4 space-y-6">
          <section class="glass-panel rounded-3xl p-6 shadow-2xl sticky top-8 flex flex-col h-auto max-h-[calc(100vh-4rem)]">
            <div class="absolute inset-0 bg-gradient-to-b from-purple-500/5 to-transparent pointer-events-none"></div>
            
            <header class="flex items-center justify-between mb-6 pb-4 border-b border-slate-800/80 relative z-10">
              <h2 class="text-xl font-bold text-slate-100 flex items-center gap-2">
                <span class="bg-purple-500 text-slate-950 w-6 h-6 rounded-full flex items-center justify-center text-sm font-bold shadow-lg shadow-purple-500/30">3</span>
                Your Playlist
              </h2>
              <span class="bg-slate-800 text-slate-300 text-xs font-bold px-2.5 py-1.5 rounded-lg shadow-inner">{{ playlist.length }} items</span>
            </header>

            <div class="flex-1 overflow-y-auto custom-scrollbar mb-6 -mr-2 pr-2 relative z-10">
              <TransitionGroup name="fade" tag="ul" class="space-y-3">
                <li v-for="track in playlist" :key="track.track_id" class="flex items-center gap-3 group bg-slate-900/60 p-3 rounded-2xl border border-slate-800/80 hover:border-slate-600 hover:bg-slate-800/80 transition-all duration-300 shadow-sm hover:shadow-md">
                  <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-slate-800 to-slate-900 flex items-center justify-center shrink-0 shadow-inner border border-slate-700/50 overflow-hidden relative group/cover">
                    <img v-if="spotifyData[track.track_id]?.cover" :src="spotifyData[track.track_id].cover" class="absolute inset-0 w-full h-full object-cover transition-opacity duration-300" :class="spotifyData[track.track_id]?.preview_url ? 'group-hover/cover:opacity-40' : ''" />
                    <svg v-else class="w-5 h-5 text-slate-500 relative z-10" fill="currentColor" viewBox="0 0 24 24"><path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z"/></svg>
                    
                    <div v-if="spotifyData[track.track_id]?.preview_url" 
                         class="absolute inset-0 flex items-center justify-center opacity-0 group-hover/cover:opacity-100 transition-opacity bg-black/40 text-white z-20 cursor-pointer"
                         @click.stop="togglePreview(spotifyData[track.track_id]?.preview_url, $event)"
                    >
                      <svg v-if="currentPreviewUrl === spotifyData[track.track_id].preview_url" class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
                      <svg v-else class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                    </div>
                  </div>
                  <div class="flex-1 min-w-0">
                    <p class="text-sm font-bold text-slate-200 truncate group-hover:text-white transition-colors">{{ track.name }}</p>
                    <p class="text-xs text-slate-400 truncate mt-0.5 flex items-center gap-1.5">
                      <img v-if="spotifyData[track.track_id]?.artist_cover" :src="spotifyData[track.track_id].artist_cover" class="w-4 h-4 rounded-full object-cover" />
                      <span>{{ track.artists }}</span>
                    </p>
                  </div>
                  <button 
                    class="shrink-0 p-2 text-slate-500 hover:text-rose-400 hover:bg-rose-500/20 rounded-xl transition-all duration-200 opacity-0 group-hover:opacity-100"
                    @click="toggleTrack(track)"
                    title="Remove"
                  >
                    <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M6 18L18 6M6 6l12 12" /></svg>
                  </button>
                </li>
              </TransitionGroup>

              <div v-if="!playlist.length" class="h-full flex flex-col items-center justify-center text-slate-500 text-sm py-16 px-4 text-center">
                <div class="w-20 h-20 rounded-full bg-slate-900/80 flex items-center justify-center mb-5 border border-slate-800 shadow-inner">
                  <svg class="w-10 h-10 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M12 6v6m0 0v6m0-6h6m-6 0H6" /></svg>
                </div>
                <p class="text-slate-400 font-medium max-w-[200px]">Add some tracks to start crafting your custom radio station.</p>
              </div>
            </div>

            <div class="mt-auto pt-5 border-t border-slate-800/80 relative z-10">
              <button
                :disabled="!playlist.length || loading"
                class="w-full py-4 rounded-2xl font-bold text-sm tracking-wide transition-all duration-300 relative overflow-hidden group shadow-lg"
                :class="!playlist.length ? 'bg-slate-900 text-slate-600 cursor-not-allowed border border-slate-800' : 'bg-gradient-to-r from-purple-600 to-emerald-500 text-white hover:shadow-xl hover:shadow-purple-500/25 border border-transparent'"
                @click="() => recommend(false)"
              >
                <div v-if="loading" class="absolute inset-0 bg-slate-900/50 backdrop-blur-sm flex items-center justify-center gap-2 text-white">
                  <svg class="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                  Analyzing Data...
                </div>
                <span v-else class="flex items-center justify-center gap-2">
                  <svg class="w-5 h-5 transition-transform duration-300 group-hover:scale-110 group-hover:rotate-12" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                  Generate Recommendations
                </span>
                <!-- Hover effect overlay -->
                <div v-if="playlist.length && !loading" class="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300 pointer-events-none"></div>
              </button>
              
              <button 
                class="w-full mt-3 py-3 rounded-2xl font-bold text-sm tracking-wide transition-all duration-300 relative overflow-hidden group shadow-md"
                :class="!playlist.length ? 'bg-slate-900 text-slate-600 cursor-not-allowed border border-slate-800' : 'bg-slate-800 text-slate-300 hover:bg-slate-700 hover:text-white border border-slate-700'"
                @click="recommend(true)"
              >
                <div v-if="loading" class="absolute inset-0 bg-slate-900/50 backdrop-blur-sm flex items-center justify-center gap-2 text-white">
                  <svg class="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                  Working...
                </div>
                <span v-else class="flex items-center justify-center gap-2">
                  <svg class="w-5 h-5 transition-transform duration-300 group-hover:-translate-y-1" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" /></svg>
                  Discover New Artists
                </span>
              </button>
              
              <button 
                v-if="playlist.length" 
                class="w-full mt-4 py-2 text-xs font-semibold text-slate-500 hover:text-slate-300 transition-colors uppercase tracking-widest"
                @click="reset"
              >
                Clear all
              </button>
            </div>
          </section>
        </div>
      </div>
      
      <!-- Error Alert -->
      <Transition name="fade">
        <div v-if="error" class="fixed bottom-6 right-6 max-w-sm bg-rose-950/95 backdrop-blur-xl border border-rose-800/80 p-5 rounded-2xl shadow-2xl flex items-start gap-4 z-50">
          <div class="w-10 h-10 rounded-full bg-rose-900/50 flex items-center justify-center shrink-0 border border-rose-800">
            <svg class="w-5 h-5 text-rose-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
          </div>
          <div class="flex-1 mt-0.5">
            <h3 class="text-sm font-bold text-rose-400">Error Occurred</h3>
            <p class="text-xs text-rose-200/80 mt-1.5 leading-relaxed">{{ error }}</p>
          </div>
          <button @click="error = ''" class="text-rose-400 hover:text-rose-200 transition-colors shrink-0 p-1"><svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" /></svg></button>
        </div>
      </Transition>

      <!-- Recommendations Results -->
      <Transition name="fade">
        <section v-if="recommendations.length" id="recommendations-section" class="mt-16 mb-24 relative z-0">
          <div class="absolute -inset-10 bg-gradient-to-b from-purple-500/10 to-transparent blur-3xl -z-10 rounded-full pointer-events-none"></div>
          
          <header class="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6 mb-10">
            <div>
              <h2 class="text-3xl sm:text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400 mb-2">
                Recommended Tracks
              </h2>
              <p class="text-slate-400 text-sm">Based on your selected playlist signature.</p>
            </div>
            <div class="hidden sm:block h-px flex-1 bg-gradient-to-r from-purple-900/50 to-transparent mx-6"></div>
          </header>
          
          <div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <div 
              v-for="(track, i) in recommendations" 
              :key="track.track_id" 
              class="glass-panel p-5 rounded-3xl flex flex-col relative overflow-hidden group hover:border-purple-500/40 transition-all duration-500 hover:-translate-y-1 hover:shadow-2xl hover:shadow-purple-900/20"
              :style="{ animationDelay: `${i * 50}ms` }"
            >
              <div class="absolute -right-4 -top-6 text-8xl font-black text-slate-800/30 -z-10 group-hover:text-purple-900/20 transition-colors duration-500">
                {{ i + 1 }}
              </div>
              
              <div class="flex items-start justify-between mb-4">
                <div class="w-12 h-12 rounded-2xl bg-gradient-to-br from-purple-500/20 to-pink-500/20 flex items-center justify-center border border-purple-500/30 shadow-inner group-hover:from-purple-500/30 group-hover:to-pink-500/30 transition-colors overflow-hidden relative group/cover">
                  <img v-if="spotifyData[track.track_id]?.cover" :src="spotifyData[track.track_id].cover" class="absolute inset-0 w-full h-full object-cover transition-opacity duration-300" :class="spotifyData[track.track_id]?.preview_url ? 'group-hover/cover:opacity-40' : ''" />
                  <svg v-else class="w-6 h-6 text-purple-400 group-hover:text-pink-300 transition-colors relative z-10" fill="currentColor" viewBox="0 0 24 24"><path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z"/></svg>
                  
                  <div v-if="spotifyData[track.track_id]?.preview_url" 
                       class="absolute inset-0 flex items-center justify-center opacity-0 group-hover/cover:opacity-100 transition-opacity bg-black/40 text-white z-20 cursor-pointer"
                       @click.stop="togglePreview(spotifyData[track.track_id]?.preview_url, $event)"
                  >
                    <svg v-if="currentPreviewUrl === spotifyData[track.track_id].preview_url" class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
                    <svg v-else class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                  </div>
                </div>
                <div class="flex flex-col items-end">
                  <span class="text-[11px] font-mono text-purple-300 font-bold bg-purple-950/60 px-2.5 py-1 rounded-lg border border-purple-800/50 shadow-sm backdrop-blur-sm flex items-center gap-1.5">
                    <svg class="w-3 h-3 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" /></svg>
                    Score: {{ track.score?.toFixed(2) }}
                  </span>
                </div>
              </div>
              
              <div class="mb-5 flex-1">
                <h3 class="text-lg font-bold text-slate-100 mb-1.5 line-clamp-1 group-hover:text-purple-300 transition-colors">{{ track.name }}</h3>
                <p class="text-sm text-slate-400 line-clamp-1 flex items-center gap-2">
                  <img v-if="spotifyData[track.track_id]?.artist_cover" :src="spotifyData[track.track_id].artist_cover" class="w-5 h-5 rounded-full object-cover shadow-sm" />
                  <span>{{ track.artists }}</span>
                </p>
              </div>
              
              <div class="mt-auto pt-4 border-t border-slate-800/80 flex items-center justify-between">
                <span class="text-[11px] text-slate-500 truncate max-w-[130px] font-medium" :title="track.genres">{{ track.genres || 'Unknown Genre' }}</span>
                <div class="flex items-center gap-2">
                  <span
                    class="px-3 py-1.5 rounded-lg border text-[10px] font-bold uppercase tracking-widest shadow-sm"
                    :class="tierOf(track).class"
                    :title="track.placement"
                  >
                    {{ tierOf(track).label }}
                  </span>
                  <button 
                    class="w-8 h-8 rounded-lg flex items-center justify-center transition-colors"
                    :class="inPlaylist.has(track.track_id) ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-800 text-slate-400 hover:bg-cyan-500 hover:text-slate-950'"
                    @click="toggleTrack(track)"
                    :title="inPlaylist.has(track.track_id) ? 'Remove from playlist' : 'Add to playlist'"
                  >
                    <svg v-if="inPlaylist.has(track.track_id)" class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" /></svg>
                    <svg v-else class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" /></svg>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </section>
      </Transition>
      <!-- Footer -->
      <footer class="mt-20 py-8 border-t border-slate-800/80 text-center text-slate-500 text-sm">
        <p class="mb-2">
          This site has only academic purposes by <strong class="text-slate-400 font-medium">Julio Espinosa Rifel</strong> for the module <strong class="text-slate-400 font-medium">Machine Learning</strong> in the MSc in Artificial Intelligence of the University of Essex Online.
        </p>
        <p class="mb-2">
          ePortfolio: <a href="https://julioespinosa.com/" target="_blank" rel="noopener noreferrer" class="text-cyan-600 hover:text-cyan-400 transition-colors">https://julioespinosa.com/</a>
          <span class="mx-2 text-slate-700">|</span>
          GitHub Repo: <a href="https://github.com/julesrif/ML-Music-Recommender-System" target="_blank" rel="noopener noreferrer" class="text-cyan-600 hover:text-cyan-400 transition-colors">https://github.com/julesrif/ML-Music-Recommender-System</a>
        </p>
        <p class="text-xs text-slate-600">
          Frontend creation was supported by Gemini 3.1 Pro. Audio previews and album covers provided by the Deezer API.
        </p>
      </footer>
    </div>
  </div>
</template>
