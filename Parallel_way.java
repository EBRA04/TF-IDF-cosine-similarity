import java.io.*;
import java.util.*;
import java.util.concurrent.*;

public class Parallel_way {

    static class Movie {
        String title, overview;
        Movie(String title, String overview) {
            this.title = title;
            this.overview = overview;
        }
    }

    static class Result {
        Movie movie;
        double score;

        Result(Movie movie, double score) {
            this.movie = movie;
            this.score = score;
        }
    }

    static Map<String, Double> computeTF(String text) {
        Map<String, Double> tf = new HashMap<>();
        String[] words = text.toLowerCase().split("\\s+");
        for (String word : words) {
            tf.put(word, tf.getOrDefault(word, 0.0) + 1);
        }
        int len = words.length;
        for (String word : tf.keySet()) {
            tf.put(word, tf.get(word) / len);
        }
        return tf;
    }

    static Map<String, Double> computeIDF(List<Movie> movies) {
        Map<String, Double> idf = new HashMap<>();
        int N = movies.size();
        for (Movie movie : movies) {
            Set<String> seen = new HashSet<>();
            for (String word : movie.overview.toLowerCase().split("\\s+")) {
                if (seen.add(word)) {
                    idf.put(word, idf.getOrDefault(word, 0.0) + 1);
                }
            }
        }
        for (String word : idf.keySet()) {
            idf.put(word, Math.log(N / idf.get(word)));
        }
        return idf;
    }

    static Map<String, Double> computeTFIDF(String text, Map<String, Double> idf) {
        Map<String, Double> tf = computeTF(text);
        Map<String, Double> tfidf = new HashMap<>();
        for (String word : tf.keySet()) {
            double idfVal = idf.getOrDefault(word, Math.log(1 + idf.size()));
            tfidf.put(word, tf.get(word) * idfVal);
        }
        return tfidf;
    }

    static double cosineSimilarity(Map<String, Double> vec1, Map<String, Double> vec2) {
        Set<String> allWords = new HashSet<>(vec1.keySet());
        allWords.addAll(vec2.keySet());
        double dot = 0, norm1 = 0, norm2 = 0;
        for (String word : allWords) {
            double v1 = vec1.getOrDefault(word, 0.0);
            double v2 = vec2.getOrDefault(word, 0.0);
            dot += v1 * v2;
            norm1 += v1 * v1;
            norm2 += v2 * v2;
        }
        return (norm1 == 0 || norm2 == 0) ? 0 : dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    static List<Movie> loadMovies(String file) throws IOException {
        List<Movie> movies = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(file));
        br.readLine();
        String line;
        while ((line = br.readLine()) != null) {
            String[] parts = line.split(",", 2);
            if (parts.length == 2) {
                movies.add(new Movie(parts[0], parts[1]));
            }
        }
        br.close();
        return movies;
    }

    static void parallelFindBest(List<Movie> movies, String input, int numThreads) throws InterruptedException {
        BlockingQueue<Result> queue = new LinkedBlockingQueue<>();
        int chunkSize = (int) Math.ceil(movies.size() / (double) numThreads);
        List<Thread> threads = new ArrayList<>();

        Map<String, Double> idf = computeIDF(movies);
        Map<String, Double> inputVector = computeTFIDF(input, idf);

        long start = System.currentTimeMillis();

        for (int i = 0; i < numThreads; i++) {
            final int startIndex = i * chunkSize;
            final int endIndex = Math.min(movies.size(), (i + 1) * chunkSize);
            Thread t = new Thread(() -> {
                Movie bestMovie = null;
                double bestScore = -1;
                for (int j = startIndex; j < endIndex; j++) {
                    Map<String, Double> movieVector = computeTFIDF(movies.get(j).overview, idf);
                    double score = cosineSimilarity(inputVector, movieVector);
                    if (score > bestScore) {
                        bestScore = score;
                        bestMovie = movies.get(j);
                    }
                }
                try {
                    queue.put(new Result(bestMovie, bestScore));
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            });
            threads.add(t);
            t.start();
        }

        for (Thread t : threads) {
            t.join();
        }

        Result best = queue.take();
        for (int i = 1; i < numThreads; i++) {
            Result r = queue.take();
            if (r.score > best.score) {
                best = r;
            }
        }

        long end = System.currentTimeMillis();

        System.out.println("\n=== RESULT using " + numThreads + " threads ===");
        System.out.println("Time: " + (end - start) + " ms");
        System.out.println("Movie: " + best.movie.title);
        System.out.println("Overview: " + best.movie.overview);
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        List<Movie> movies = loadMovies("Movie.csv");
        System.out.println("Loaded " + movies.size() + " movies!");

        Scanner sc = new Scanner(System.in);
        System.out.print("Enter keywords: ");
        String input = sc.nextLine();

        int[] threadCounts = {4, 8, 16, 32};
        for (int threads : threadCounts) {
            parallelFindBest(movies, input, threads);
        }

        sc.close();
    }
}
