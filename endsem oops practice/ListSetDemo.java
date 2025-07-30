import java.util.*;

public class ListSetDemo {
    public static void main(String[] args) {
        ArrayList<String> list = new ArrayList<>();
        list.add("Java");
        list.add("Java");
        System.out.println("ArrayList: " + list);

        HashSet<String> set = new HashSet<>();
        set.add("Java");
        set.add("Java");
        System.out.println("HashSet: " + set);
    }
}
