<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'

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

async function recommend() {
  if (!playlist.value.length) return
  loading.value = true
  error.value = ''
  recommendations.value = []
  try {
    const res = await fetch('/api/recommend/new', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ track_ids: playlist.value.map((t) => t.track_id), k: 10 }),
    })
    if (!res.ok) throw new Error((await res.json()).detail ?? 'Request failed')
    recommendations.value = (await res.json()).recommendations
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
  <div class="min-h-screen bg-slate-950 text-slate-100 p-6 sm:p-10">
    <div class="max-w-6xl mx-auto">
      <header class="mb-8">
        <h1 class="text-3xl sm:text-4xl font-bold text-emerald-400">Playlist Continuation</h1>
        <p class="text-slate-400 mt-2 text-sm max-w-3xl">
          Pick an artist, add a few of their tracks, and get ten suggestions for what could
          come next.
        </p>
        <p v-if="stats" class="text-slate-500 mt-1 text-xs">
          {{ stats.tracks.toLocaleString() }} tracks by
          {{ stats.artists.toLocaleString() }} artists. Each track carries a badge showing
          whether it was learned from listening data or placed from its artist.
        </p>
      </header>

      <div class="grid gap-6 lg:grid-cols-3">
        <section class="bg-slate-900 rounded-xl p-5 border border-slate-800">
          <h2 class="font-semibold mb-3">1. Pick an artist</h2>

          <div class="flex items-center justify-between mb-1">
            <span class="text-xs text-slate-500">Filter by genre</span>
            <button
              v-if="selectedGenres.length"
              class="text-xs text-emerald-500 hover:text-emerald-400"
              @click="selectedGenres = []"
            >
              Clear {{ selectedGenres.length }}
            </button>
          </div>
          <div class="flex flex-wrap gap-1 max-h-32 overflow-y-auto mb-3 pr-1">
            <button
              v-for="genre in genres"
              :key="genre.name"
              class="px-2 py-0.5 rounded-full border text-[11px] transition-colors"
              :class="
                selectedGenres.includes(genre.name)
                  ? 'bg-emerald-600 border-emerald-500 text-white'
                  : 'bg-slate-950 border-slate-700 text-slate-400 hover:border-slate-500'
              "
              :title="`${genre.artist_count.toLocaleString()} artists`"
              @click="toggleGenre(genre.name)"
            >
              {{ genre.name }}
            </button>
          </div>

          <input
            v-model="artistQuery"
            type="text"
            placeholder="Search artists..."
            class="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-700 focus:border-emerald-500 focus:outline-none text-sm"
          />
          <p class="text-xs text-slate-500 mt-2 flex items-center justify-between gap-2">
            <span>
              {{
                loadingArtists
                  ? 'Loading...'
                  : `${artistTotal.toLocaleString()} ${artistTotal === 1 ? 'artist' : 'artists'}`
              }}
            </span>
            <span class="shrink-0 inline-flex rounded-md border border-slate-700 overflow-hidden">
              <button
                v-for="option in [
                  { key: 'plays', label: 'Popular' },
                  { key: 'name', label: 'A-Z' },
                ]"
                :key="option.key"
                class="px-2 py-0.5 transition-colors"
                :class="
                  artistSort === option.key
                    ? 'bg-emerald-600 text-white'
                    : 'text-slate-400 hover:text-slate-200'
                "
                @click="artistSort = option.key as 'plays' | 'name'"
              >
                {{ option.label }}
              </button>
            </span>
          </p>

          <ul
            class="mt-2 max-h-96 overflow-y-auto divide-y divide-slate-800"
            @scroll="onArtistScroll"
          >
            <li v-for="artist in artists" :key="artist.name">
              <button
                class="w-full text-left py-2 px-2 rounded hover:bg-slate-800 flex items-center gap-2"
                :class="selectedArtist?.name === artist.name ? 'bg-slate-800' : ''"
                @click="openArtist(artist)"
              >
                <span class="truncate flex-1 text-sm">{{ artist.name }}</span>
                <span class="shrink-0 text-xs text-slate-500">{{ artist.track_count }}</span>
              </button>
            </li>
            <li v-if="loadingMoreArtists" class="py-2 text-center text-xs text-slate-500">
              Loading more...
            </li>
          </ul>
          <p v-if="!loadingArtists && !artists.length" class="text-sm text-amber-400/90 mt-3">
            No artist matches
            {{ selectedGenres.length ? 'that search and genre selection' : 'that search' }}.
          </p>
        </section>

        <section class="bg-slate-900 rounded-xl p-5 border border-slate-800">
          <h2 class="font-semibold mb-3">2. Add tracks</h2>
          <input
            v-model="trackQuery"
            type="text"
            placeholder="Search any track by title or artist..."
            class="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-700 focus:border-emerald-500 focus:outline-none text-sm"
          />
          <p class="text-xs text-slate-500 mt-2">
            <template v-if="searchingTracks || loadingTracks">Loading...</template>
            <template v-else-if="searchMode">
              {{ trackTotal.toLocaleString() }} {{ trackTotal === 1 ? 'track' : 'tracks' }} found
              <button class="text-emerald-500 hover:text-emerald-400" @click="trackQuery = ''">
                - clear
              </button>
            </template>
            <template v-else-if="selectedArtist">
              {{ artistTracks.length }}
              {{ artistTracks.length === 1 ? 'track' : 'tracks' }} by {{ selectedArtist.name }}
            </template>
            <template v-else>Pick an artist on the left, or search here.</template>
          </p>

          <ul class="mt-2 max-h-96 overflow-y-auto divide-y divide-slate-800">
            <li v-for="track in visibleTracks" :key="track.track_id" class="py-2 flex items-center gap-3">
              <div class="min-w-0 flex-1">
                <p class="truncate text-sm">{{ track.name }}</p>
                <div class="flex items-center gap-2 min-w-0">
                  <span
                    class="shrink-0 px-1.5 py-0.5 rounded border text-[10px] font-medium"
                    :class="tierOf(track).class"
                    :title="track.placement"
                  >
                    {{ tierOf(track).label }}
                  </span>
                  <p class="truncate text-xs text-slate-500">
                    {{ searchMode ? track.artists : track.genres }}
                  </p>
                </div>
              </div>
              <button
                class="shrink-0 px-2 py-1 rounded-md text-xs font-semibold"
                :class="
                  inPlaylist.has(track.track_id)
                    ? 'bg-slate-700 text-slate-300'
                    : 'bg-emerald-600 hover:bg-emerald-500'
                "
                @click="toggleTrack(track)"
              >
                {{ inPlaylist.has(track.track_id) ? 'Remove' : 'Add' }}
              </button>
            </li>
          </ul>

          <p
            v-if="searchMode && !searchingTracks && !trackResults.length"
            class="text-sm text-amber-400/90 mt-3"
          >
            No track matches that search.
          </p>
        </section>

        <section class="bg-slate-900 rounded-xl p-5 border border-slate-800">
          <div class="flex items-center justify-between mb-3">
            <h2 class="font-semibold">3. Your playlist ({{ playlist.length }})</h2>
            <button v-if="playlist.length" class="text-xs text-slate-400 hover:text-slate-200" @click="reset">
              Clear
            </button>
          </div>

          <p v-if="!playlist.length" class="text-sm text-slate-500">
            Add tracks from any number of artists.
          </p>

          <ul v-else class="max-h-80 overflow-y-auto divide-y divide-slate-800 mb-4">
            <li v-for="track in playlist" :key="track.track_id" class="py-2 flex items-center gap-2">
              <div class="min-w-0 flex-1">
                <p class="truncate text-sm">{{ track.name }}</p>
                <p class="truncate text-xs text-slate-500">{{ track.artists }}</p>
              </div>
              <button class="shrink-0 text-slate-500 hover:text-rose-400 text-lg leading-none" @click="toggleTrack(track)">
                ×
              </button>
            </li>
          </ul>

          <button
            :disabled="!playlist.length || loading"
            class="w-full py-3 rounded-lg font-semibold bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 disabled:text-slate-600"
            @click="recommend"
          >
            {{ loading ? 'Scoring catalogue...' : 'Continue this playlist' }}
          </button>
        </section>
      </div>

      <p v-if="error" class="mt-6 px-4 py-3 rounded-lg bg-rose-950 border border-rose-800 text-rose-200 text-sm">
        {{ error }}
      </p>

      <section v-if="recommendations.length" class="mt-8 bg-slate-900 rounded-xl p-5 border border-slate-800">
        <h2 class="font-semibold mb-4 text-emerald-400">Recommended next tracks</h2>
        <ol class="divide-y divide-slate-800">
          <li v-for="(track, i) in recommendations" :key="track.track_id" class="py-3 flex items-center gap-4">
            <span class="w-6 text-slate-600 font-mono text-sm">{{ i + 1 }}</span>
            <div class="min-w-0 flex-1">
              <p class="truncate font-medium">{{ track.name }}</p>
              <div class="flex items-center gap-2 min-w-0">
                <span
                  class="shrink-0 px-1.5 py-0.5 rounded border text-[10px] font-medium"
                  :class="tierOf(track).class"
                  :title="track.placement"
                >
                  {{ tierOf(track).label }}
                </span>
                <p class="truncate text-xs text-slate-400">{{ track.artists }}</p>
              </div>
            </div>
            <span class="shrink-0 text-xs text-slate-500 hidden sm:block">{{ track.genres }}</span>
            <span class="shrink-0 font-mono text-xs text-emerald-500">{{ track.score?.toFixed(1) }}</span>
          </li>
        </ol>
      </section>
    </div>
  </div>
</template>
